import numpy
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(numpy.float32))
X_test = torch.from_numpy(X_test.astype(numpy.float32))
y_train = torch.from_numpy(y_train.astype(numpy.float32))
y_test = torch.from_numpy(y_test.astype(numpy.float32))

y_train =y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0], 1)


# 1) model
# f = wx + b
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model =LogisticRegression(n_features)


# 2) loss and optimizer
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward bass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backwards pass
    loss.backward()

    # update
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    if(epoch+1) % 10 ==0:
        print(f"epoch: {epoch+1}, loss= {loss.item():.4f}")

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"accuracy ={accuracy:.4f}")