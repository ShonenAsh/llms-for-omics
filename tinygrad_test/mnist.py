"""
    Usage: python mnist.py

    Reference written by Ashish M. for testing LLMs
"""


from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Adam 
from tinygrad import Context
from tinygrad import TinyJit


class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))


def main():
    X_train, Y_train, X_test, Y_test = mnist()
    print(f"{X_train.shape}, {Y_train.shape}, {X_test.shape}, {Y_test.shape}")
    
    model = Model()

    optim = Adam(nn.state.get_parameters(model))
    batch_size = 128
    

    def step():
        Tensor.training = True  # unfreeze dropout layers
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        X, Y = X_train[samples], Y_train[samples]
        optim.zero_grad()
        loss = model(X).sparse_categorical_crossentropy(Y).backward()
        optim.step()
        return loss 
    
    jit_step = TinyJit(step)

    with Context():
      for st in range(7000):
            loss = jit_step()
            if st %100 == 0:
                Tensor.training = False
                acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
                print(f"step {st:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
    
if __name__ == "__main__":
    main()
