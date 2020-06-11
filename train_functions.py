
import time
from torch.backends import cudnn
#Some functions for training

def train_model(net, dataloader, criterion, scheduler, optimizer, num_epochs = 10, log_frequency = 10):
    DEVICE = "cuda"
    since = time.time()
    cudnn.benchmark
    net = net.to(DEVICE)
    current_step = 0
    loss_array = []
    step_array = []
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}, LR = {}'.format(epoch+1, num_epochs,
                                                     scheduler.get_last_lr()))

        for images,labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            net.train()
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)


            if current_step % log_frequency == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))
                loss_array.append(loss)
                step_array.append(current_step)

            loss.backward()
            optimizer.step()
            current_step += 1

        accuracy = test_model(net, art_dl, art_dataset)
        accuracy_array.append(accuracy)
        epochs_array.append(epoch)
        scheduler.step()

    plt.plot(step_array,loss_array)
    plt.xlabel("Step number")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(epochs_array, accuracy_array)
    plt.xlabel("Epochs number")
    plt.ylabel("Accuracy")
    plt.show()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return net

def test_model(net, dataloader, dataset):
    DEVICE = 'cuda'
    running_corrects = 0
    for images, labels in tqdm(dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = net(images)

        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = running_corrects/float(len(dataset))
    print('Test Accuracy: {}'.format(accuracy))
    return accuracy

##TRAIN ON PHOTO AND TEST ON ART WITH DANN ADAPTATION
def train_model_with_dann(net, dataloader_src, dataloader_target, criterion, scheduler, optimizer, num_epochs = 10, batch_size = 128, alphav = 0.01, log_frequency = 10):
    DEVICE = "cuda"
    since = time.time()
    cudnn.benchmark
    net = net.to(DEVICE)
    current_step = 0

    zeros = np.zeros(batch_size)
    zero_target = torch.as_tensor(zeros).long()
    zero_target = zero_target.to(DEVICE)
    ones = np.ones(batch_size)
    one_target = torch.as_tensor(ones).long()
    one_target = one_target.to(DEVICE)

    for epoch in range(num_epochs):
        print('Starting epoch {}/{}, LR = {}'.format(epoch+1, num_epochs,
                                                     scheduler.get_last_lr()))

        for i in range(len(dataloader_src)):
            images, labels = next(iter(dataloader_src))
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            net.train()
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            outputs = net(images, alpha = alphav)
            loss = criterion(outputs, zero_target)
            loss.backward()

            target_images, target_labels = next(iter(dataloader_target))

            target_images = target_images.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            outputs = net(target_images, alpha = alphav)
            loss = criterion(outputs, one_target)


            if current_step % log_frequency == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))

            loss.backward()
            optimizer.step()
            current_step += 1

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("alpha: " + str(alpha))

    return net
