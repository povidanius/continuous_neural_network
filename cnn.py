import torch
import torch.nn as nn

class alpha(nn.Module):
    def __init__(self):
        super(alpha, self).__init__()
        self.w1 = nn.Linear(1,8)
        self.w2 = nn.Linear(8,1)
        self.sigma = nn.Sigmoid()
    def forward(self,t):
        return self.w2(self.sigma(self.w1(t)))


class W(nn.Module):
    def __init__(self, dx):
        super(W, self).__init__()
        self.w1 = nn.Linear(1,8)
        self.w2 = nn.Linear(8, dx)
        self.sigma = nn.Sigmoid()
    def forward(self,t):
        return self.w2(self.sigma(self.w1(t)))


class ContinousNeuralNetwork(nn.Module):
    def __init__(self, dx):
        super(ContinousNeuralNetwork, self).__init__()

        self.alpha = alpha()
        self.W = W(dx)
        self.act = nn.Sigmoid()
        self.t = torch.range(0,1,1/1000).unsqueeze(1)


    def forward(self, x):

        alpha_val_t = self.alpha(self.t)
        w_val_t = self.W(self.t)
        wx_t = self.act(torch.matmul(w_val_t, x.t()))
        ff_t = alpha_val_t * wx_t
        o = torch.trapz(ff_t,self.t, dim=0)
        return o

    


if __name__ == "__main__":
    dim = 16
    x = torch.randn(5, dim)
    model = ContinousNeuralNetwork(dim)
    y = model(x)
    print(y)
    breakpoint()