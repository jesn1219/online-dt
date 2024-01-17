
import transformers
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import torch
import torch.nn as nn
import transformers
import numpy as np
from decision_transformer.models.model import TrajectoryModel01
from customs.stochastic_policy import DiagGaussianActor


class DecisionTransformer02(TrajectoryModel01):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    """
    jesnk: this model uses gpt to model (state_1, state_2, ..., state_n)
    
    """
    def __init__(
        self,
        state_dim,
        hidden_size,
        state_range,
        ordering=0,
        act_dim = None,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        state_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        state_mean=None, #jesnk
        state_std=None, #jesnk
        **kwargs
    ):
        super().__init__(state_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        if ordering:
            self.embed_ordering = nn.Embedding(max_ep_len, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        if stochastic_policy:
            self.predict_state = DiagGaussianActor(hidden_size, self.state_dim)
        else:
            self.predict_state = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.state_dim)]
                    + ([nn.Tanh()] if state_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy
        self.eval_context_length = eval_context_length
        self.ordering = ordering
        self.state_range = state_range
        #self.state_mean = state_mean # jesnk
        #self.state_std = state_std # jesnk
        #print(f'jesnk: debug: DT01: state_mean:{self.state_mean}, state_std:{self.state_std}')


        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        timesteps=None,
        ordering=None,
        padding_mask=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1] # 512, seq_legnth, 

        if padding_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            padding_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        if timesteps is None :
            timesteps = torch.arange(seq_length, device=states.device).repeat(batch_size, 1)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)

        if self.ordering:
            order_embeddings = self.embed_ordering(timesteps)
        else:
            order_embeddings = 0.0

        state_embeddings = state_embeddings + order_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # state_embeddings.shape: batch, seq_length, hidden_size
        stacked_inputs = (
            #torch.stack((state_embeddings), dim=1)
            state_embeddings # batch, seq_length, hidden_size
            #.permute(0, 2, 1, 3) # batch, seq_length, 1, hidden_size
            .reshape(batch_size, 1 * seq_length, self.hidden_size)
        )
        # stacked_inputs.shape: batch, 1*seq_length, hidden_size
        stacked_inputs = self.embed_ln(stacked_inputs) #jesnk: layer norm disabled

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_padding_mask = (
            #torch.stack((padding_mask, padding_mask, padding_mask), dim=1)
            padding_mask
            #.permute(0, 2, 1)
            .reshape(batch_size, 1 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_padding_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 1, self.hidden_size).permute(0, 2, 1, 3)
        # after reshape : batch, 1, seq_length, hidden_size
        # get predictions
        # predict next state given state and action

        #state_preds = self.predict_state(x[:, 0]) # jesnk: DT1 setting
        states_pred = self.predict_state(x[:,0]) # jesnk: must check the index of [:, 0] is correct

        # state_preds.shape: batch, seq_length, state_dim
        return states_pred

    def get_predictions(
        self, states, timesteps, num_envs=1, **kwargs
    ):
        # we don't care about the past rewards in this model
        # tensor shape: batch_size, seq_length, variable_dim
        states = states.reshape(num_envs, -1, self.state_dim)

        # tensor shape: batch_size, seq_length
        timesteps = timesteps.reshape(num_envs, -1)

        # max_length is the DT context length (should be input length of the subsequence)
        # eval_context_length is the how long you want to use the history for your prediction
        if self.max_length is not None:
            states = states[:, -self.eval_context_length :]
            timesteps = timesteps[:, -self.eval_context_length :]

            ordering = torch.tile(
                torch.arange(timesteps.shape[1], device=states.device),
                (num_envs, 1),
            )
            # pad all tokens to sequence length
            padding_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            padding_mask = padding_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            padding_mask = padding_mask.repeat((num_envs, 1))

            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)

            ordering = torch.cat(
                [
                    torch.zeros(
                        (ordering.shape[0], self.max_length - ordering.shape[1]),
                        device=ordering.device,
                    ),
                    ordering,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            padding_mask = None

        state_preds = self.forward(
            states,
            timesteps,
            ordering,
            padding_mask=padding_mask,
            **kwargs
        )
        if self.stochastic_policy:
            return state_preds
        else:
            return (
                self.clamp_state(state_preds[:, -1])
            )

    def clamp_state(self, state):
        return state.clamp(*self.state_range)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
