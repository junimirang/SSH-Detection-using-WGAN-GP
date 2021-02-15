import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import pandas as pd


def loading_dataset(): ## After loading csv file, Pandas Data Frameset Generation ##
    # time.taken   c.ip   response.code  response.type  sc.byte    cs.byte    method URI    cs.host    Destination Port
    # cs_user_agent    sc_filter_result   category   Destination isp    region no_url
    # ratio_trans_receive  count_total_connect    count_connect_IP
    # log_time_taken   log_cs_byte    log_ratio_trans_receive    log_count_connect_IP
    # log_count_total_connect  avg_count_connect  log_avg_count_connect  transmit_speed_BPS
    # log_transmit_speed_BPS   LABEL
    df = pd.read_csv('df_training_20201110.csv', index_col=0)
    #df_compare = pd.read_csv('Flow2Session/df_compare_20201110.csv', index_col=0)
    X = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    Y = df[["LABEL"]]
    Z = df[["log_time_taken"]]
    K = df[["no_url"]]
    L = df[["log_ratio_trans_receive"]]
    N = df[["Destination", "Destination Port", "no_url"]]
    GAN_dataset = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Business.time","log_time_taken","no_url","log_ratio_trans_receive","Destination", "Destination Port"]]
    #X_compare = df_compare[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    #Y_compare = df_compare[["LABEL"]]
    #Z_compare = df_compare[["log_time_taken"]]
    #K_compare = df_compare[["no_url"]]
    #L_compare = df_compare[["log_ratio_trans_receive"]]
    #N_compare = df_compare[["Destination", "Destination Port", "no_url"]]
    #return(X, K, L, Z, Y, N, X_compare, K_compare, L_compare, Z_compare, Y_compare, N_compare)
    return(GAN_dataset)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, x):
        output = self.model(x)
        return output

if __name__ == '__main__':

    pd.set_option('display.max_rows', 100)     # Maximum rows for print
    pd.set_option('display.max_columns', 20)   # Maximum columns for print
    pd.set_option('display.width', 20)         # Maximum witch for print
    GAN_dataset = loading_dataset()

    ## PANDAS to Tensor : https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
    

    torch.manual_seed(111)
    train_data_length = 1024
    train_data = torch.zeros((train_data_length, 2))
    temp = torch.rand(train_data_length)
    train_data[:, 0] = 2 * math.pi * temp
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (train_data[i], train_labels[i]) for i in range(train_data_length)
    ]
    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    plt.show()
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    discriminator = Discriminator()
    generator = Generator()
    lr = 0.001 ## learning rate
    num_epochs = 300 ## number of ephchs
    loss_function = nn.BCELoss() ## assigns the variable loss_function to the binary cross-entropy function BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2))
            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()
            # Show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show()
    print("pause")
