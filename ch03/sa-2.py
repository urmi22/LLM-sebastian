# same as sa-1 except that, instead of nn.Parameter we use nn.Linear layer
'''
because, nn.Linear effectively perform matrix multiplication when the bias units are disabled. 
Additionally, a significant advantage of using nn.Linear is instead of manually implementing nn.Parameter(torch.rand(...))
nn.Linear  has an optimized weight initialization scheme, contributing to more stable and effective model training.

'''