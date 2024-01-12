class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, train_loader, epochs=10):
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                # 正向傳播
                # predictions = np.expm1(self.model(inputs).detach().numpy())
                # print(inputs)
                predictions = self.model(inputs)
                # print('prediction: ', predictions, ', target: ', targets)
                loss = self.loss_fn(predictions, targets)

                # 反向傳播和優化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                print('prediction: ', predictions,
                      ', target: ', targets,
                      ', loss item: ', loss.item(),
                      ', total lose: ', total_loss)

            average_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')
