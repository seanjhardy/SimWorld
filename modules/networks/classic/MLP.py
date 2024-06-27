from torch import nn


class MLP(nn.Module):

    def __init__(self, input_size, n_embed=None, output_size=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = output_size or input_size
        hidden_features = n_embed or input_size
        drop_probs = (drop, drop) if isinstance(drop, float) else drop

        self.fc1 = nn.Linear(input_size, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop2(self.fc2(x))
        return x
