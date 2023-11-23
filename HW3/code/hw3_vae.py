import sys
import argparse
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
import matplotlib.image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from hw3_utils import array_to_image, concat_images, batch_indices, load_mnist




# The "encoder" model q(z|x)
class Encoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Encoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units
        
        self.fc1 = nn.Linear(data_dimension, hidden_units)
        self.fc2_mu = nn.Linear(hidden_units, latent_dimension)
        self.fc2_sigma = nn.Linear(hidden_units, latent_dimension)

    def forward(self, x):
        # Input: x input image [batch_size x data_dimension]
        # Output: parameters of a diagonal gaussian 
        #   mean : [batch_size x latent_dimension]
        #   variance : [batch_size x latent_dimension]

        hidden = torch.tanh(self.fc1(x))
        mu = self.fc2_mu(hidden)
        log_sigma_square = self.fc2_sigma(hidden)
        sigma_square = torch.exp(log_sigma_square)  
        return mu, sigma_square


# "decoder" Model p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dimension, hidden_units, data_dimension):
        super(Decoder, self).__init__()
        # Input:
        #   latent_dimension: the latent dimension of the encoder
        #   hidden_units: the number of hidden units

        # TODO: define the parameters of the decoder
        self.fc1 = nn.Linear(latent_dimension, hidden_units)
        self.fc2 = nn.Linear(hidden_units,data_dimension)
        # fc1: a fully connected layer with 500 hidden units. 
        # fc2: a fully connected layer with 500 hidden units. 

    def forward(self, z):
        # input
        #   z: latent codes sampled from the encoder [batch_size x latent_dimension]
        # output 
        #   p: a tensor of the same size as the image indicating the probability of every pixel being 1 [batch_size x data_dimension]

        # TODO: implement the decoder here. The decoder is a multi-layer perceptron with two hidden layers. 
        # The first layer is followed by a tanh non-linearity and the second layer by a sigmoid.

        layer1 = torch.tanh(self.fc1(z))
        p = torch.sigmoid(self.fc2(layer1))
        
        return p


# VAE model
class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.latent_dimension = args.latent_dimension
        self.hidden_units =  args.hidden_units
        self.data_dimension = args.data_dimension
        self.resume_training = args.resume_training
        self.batch_size = args.batch_size
        self.num_epoches = args.num_epoches
        self.e_path = args.e_path
        self.d_path = args.d_path

        # load and pre-process the data
        N_data, self.train_images, self.train_labels, test_images, test_labels = load_mnist()


        # Instantiate the encoder and decoder models 
        self.encoder = Encoder(self.latent_dimension, self.hidden_units, self.data_dimension)
        self.decoder = Decoder(self.latent_dimension, self.hidden_units, self.data_dimension)

        # Load the trained model parameters
        if self.resume_training:
            self.encoder.load_state_dict(torch.load(self.e_path))
            self.decoder.load_state_dict(torch.load(self.d_path))

    # Sample from Diagonal Gaussian z~N(μ,σ^2 I) 
    @staticmethod
    def sample_diagonal_gaussian(mu, sigma_square):
        # Inputs:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   sample: from a diagonal gaussian with mean mu and variance sigma_square [batch_size x latent_dimension]

        # TODO: Implement the reparameterization trick and return the sample z [batch_size x latent_dimension]
        sample = mu + torch.sqrt(sigma_square) * torch.normal(mean = 0, std = 1, size = sigma_square.shape)
        return sample

    # Sampler from Bernoulli
    @staticmethod
    def sample_Bernoulli(p):
        # Input: 
        #   p: the probability of pixels labeled 1 [batch_size x data_dimension]
        # Output:
        #   x: pixels'labels [batch_size x data_dimension], type should be torch.float32

        # TODO: Implement a sampler from a Bernoulli distribution
        x = torch.rand(p.shape)
        x = (x < p).float().to(p.device)
        return x


    # Compute Log-pdf of z under Diagonal Gaussian N(z|μ,σ^2 I)
    @staticmethod
    def logpdf_diagonal_gaussian(z, mu, sigma_square):
        # Input:
        #   z: sample [batch_size x latent_dimension]
        #   mu: mean of the gaussian distribution [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian distribution [batch_size x latent_dimension]
        # Output:
        #    logprob: log-probability of a diagomnal gaussian [batch_size]
        
        # TODO: implement the logpdf of a gaussian with mean mu and variance sigma_square*I
        log2pi = torch.log(torch.tensor(2 * np.pi, device=z.device, dtype=z.dtype))
        log_likelihood = -0.5 * (log2pi + torch.log(sigma_square) + (z - mu)**2 / sigma_square)
        return torch.sum(log_likelihood,dim=-1)

    # Compute log-pdf of x under Bernoulli 
    @staticmethod
    def logpdf_bernoulli(x, p):
        # Input:
        #   x: samples [batch_size x data_dimension]
        #   p: the probability of the x being labeled 1 (p is the output of the decoder) [batch_size x data_dimension]
        # Output:
        #   logprob: log-probability of a bernoulli distribution [batch_size]

        # TODO: implement the log likelihood of a bernoulli distribution p(x)
        logprob = x*torch.log(p) + (1-x) * torch.log(1-p)
        return torch.sum(logprob,dim=-1)
    
    # Sample z ~ q(z|x)
    def sample_z(self, mu, sigma_square):
        # input:
        #   mu: mean of the gaussian [batch_size x latent_dimension]
        #   sigma_square: variance of the gaussian [batch_size x latent_dimension]
        # Output:
        #   zs: samples from q(z|x) [batch_size x latent_dimension] 
        zs = self.sample_diagonal_gaussian(mu, sigma_square)
        return zs 


    # Variational Objective
    def elbo_loss(self, sampled_z, mu, sigma_square, x, p):
        # Inputs
        #   sampled_z: samples z from the encoder [batch_size x latent_dimension]
        #   mu:
        #   sigma_square: parameters of q(z|x) [batch_size x latent_dimension]
        #   x: data samples [batch_size x data_dimension]
        #   p: the probability of a pixel being labeled 1 [batch_size x data_dimension]
        # Output
        #   elbo: the ELBO loss (scalar)

        # log_q(z|x) logprobability of z under approximate posterior N(μ,σ)
        log_q = self.logpdf_diagonal_gaussian(sampled_z, mu, sigma_square)
        
        # log_p_z(z) log probability of z under prior
        z_mu = torch.FloatTensor([0]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        z_sigma = torch.FloatTensor([1]*self.latent_dimension).repeat(sampled_z.shape[0], 1)
        log_p_z = self.logpdf_diagonal_gaussian(sampled_z, z_mu, z_sigma)

        # log_p(x|z) - conditional probability of data given latents.
        log_p = self.logpdf_bernoulli(x, p)
        
        # TODO: implement the ELBO loss using log_q, log_p_z and log_p
        elbo = log_p + log_p_z - log_q
        
        return torch.mean(elbo)


    def train(self):
        
        # Set-up ADAM optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        adam_optimizer = optim.Adam(params)

        # Train for ~200 epochs 
        num_batches = int(np.ceil(len(self.train_images) / self.batch_size))
        num_iters = self.num_epoches * num_batches
        
        for i in range(num_iters):
            x_minibatch = self.train_images[batch_indices(i, num_batches, self.batch_size),:]
            adam_optimizer.zero_grad()

            mu, sigma_square = self.encoder(x_minibatch)
            zs = self.sample_z(mu, sigma_square)
            p = self.decoder(zs)
            elbo = self.elbo_loss(zs, mu, sigma_square, x_minibatch, p)
            total_loss = -elbo
            total_loss.backward()
            adam_optimizer.step()

            if i%100 == 0:
                print("Epoch: " + str(i//num_batches) + ", Iter: " + str(i) + ", ELBO:" + str(elbo.item()))

        # Save Optimized Model Parameters
        torch.save(self.encoder.state_dict(), self.e_path)
        torch.save(self.decoder.state_dict(), self.d_path)


    # Generate digits using the VAE
    def visualize_data_space(self):
        # TODO: Sample 10 z from prior 
        z = torch.normal(0,1,size=(10,self.latent_dimension))

        # TODO: For each z, plot p(x|z)
        imgs = []
        for i in (self.decoder(z)):
            p_x_z = array_to_image(i.detach().numpy())
            
            imgs.append(p_x_z)
            imgs.append(array_to_image(self.sample_Bernoulli(i).detach().numpy()))
        out = concat_images(imgs, 2, 10)
        print(type(out))
        plt.imshow(out, cmap="gray")
        plt.show()

            


        # TODO: Sample x from p(x|z) 
        
        

        # TODO: Concatenate plots into a figure (use the function concat_images)
        

        # TODO: Save the generated figure and include it in your report
        
        
    # Produce a scatter plot in the latent space, where each point in the plot will be the mean vector 
    # for the distribution $q(z|x)$ given by the encoder. Further, we will colour each point in the plot 
    # by the class label for the input data. Each point in the plot is colored by the class label for 
    # the input data.
    # The latent space should have learned to distinguish between elements from different classes, even though 
    # we never provided class labels to the model!
    def visualize_latent_space(self):
        
        # TODO: Encode the training data self.train_images
        mu, sigma_squared = self.encoder(self.train_images)

        # TODO: Take the mean vector of each encoding
        mu_np = mu.detach().numpy()
        print(mu_np.shape)
        print(self.train_labels.detach().numpy().shape)
        class_labels = np.argmax(self.train_labels.detach().numpy(),axis=1)
        scatter = plt.scatter(mu_np[:,0],mu_np[:,1], c = class_labels, cmap="tab10")
        # for color in range(10):
        #     rows = class_labels == color
        #     plt.scatter(mu_np[rows,0],mu_np[rows,1], c = class_labels[rows], label = color)
        plt.title("Latent Space Visualization")
        plt.legend(*scatter.legend_elements(),loc='center right', bbox_to_anchor=(1.12, 0.5), title="Classes")
        plt.show()

        

        # TODO: Plot these mean vectors in the latent space with a scatter
        # Colour each point depending on the class label 
        

        # TODO: Save the generated figure and include it in your report
        
        


    # Function which gives linear interpolation z_α between za and zb
    @staticmethod
    def interpolate_mu(mua, mub, alpha = 0.5):
        return alpha*mua + (1-alpha)*mub


    # A common technique to assess latent representations is to interpolate between two points.
    # Here we will encode 3 pairs of data points with different classes.
    # Then we will linearly interpolate between the mean vectors of their encodings. 
    # We will plot the generative distributions along the linear interpolation.
    def visualize_inter_class_interpolation(self):
        
        # TODO: Sample 3 pairs of data with different classes
        class_labels = np.argmax(self.train_labels.detach().numpy(),axis=1)
        row1 = class_labels == 1
        row2 = class_labels == 2
        row5 = class_labels == 5
        row7 = class_labels == 7
        row8 = class_labels == 8
        row0 = class_labels == 0
        print(self.train_images.shape)
        x1 = self.train_images[row1][1,:]
        x2 = self.train_images[row2][1,:]
        x5 = self.train_images[row5][1,:]
        x7 = self.train_images[row7][1,:]
        x8 = self.train_images[row8][1,:]
        x0 = self.train_images[row0][1,:]
        print(x1.shape)
        # TODO: Encode the data in each pair, and take the mean vectors
        mu1, s1 = self.encoder(x1)
        mu2, s2 = self.encoder(x2)
        mu5, s5 = self.encoder(x5)
        mu7, s7 = self.encoder(x7)
        mu8, s8 = self.encoder(x8)
        mu0, s0 = self.encoder(x0)
        


        # TODO: Linearly interpolate between these mean vectors (Use the function interpolate_mu)
        images = []
        for i in range(0,11):
            za1 = self.interpolate_mu(mu1, mu2, i/10)
            za2 = self.interpolate_mu(mu5, mu7, i/10)
            za3 = self.interpolate_mu(mu8, mu0, i/10)
            images.append(array_to_image(self.decoder(za1).detach().numpy()))
            images.append(array_to_image(self.decoder(za2).detach().numpy()))
            images.append(array_to_image(self.decoder(za3).detach().numpy()))
        
        
        
        out = concat_images(images,3,10)
        plt.imshow(out, cmap="gray")
        plt.show()

        # TODO: Along the interpolation, plot the distributions p(x|z_α)
        

        # Concatenate these plots into one figure
        pass
        
        
      

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--e_path', type=str, default="./e_params.pkl", help='Path to the encoder parameters.')
    parser.add_argument('--d_path', type=str, default="./d_params.pkl", help='Path to the decoder parameters.')
    parser.add_argument('--hidden_units', type=int, default=500, help='Number of hidden units of the encoder and decoder models.')
    parser.add_argument('--latent_dimension', type=int, default='2', help='Dimensionality of the latent space.')
    parser.add_argument('--data_dimension', type=int, default='784', help='Dimensionality of the data space.')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--num_epoches', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')

    args = parser.parse_args()
    return args


def main():
    
    # read the function arguments
    args = parse_args()

    # set the random seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
   
        
    
    # train the model 
    vae = VAE(args)
    vae.train()

    # visualize the latent space
    vae.visualize_data_space()
    vae.visualize_latent_space()
    vae.visualize_inter_class_interpolation()


main()