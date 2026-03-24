from torch import nn


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 model execution is not implemented yet. "
            "This registration only enables config/model selection."
        )

    def load_weights(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepSeek-V2 weight loading is not implemented yet. "
            "The MLA attention/model path still needs to be added."
        )
