import torch
import torch.nn as nn


class conv_block(nn.Module):
    '''
    Convolution blocks of UNet
    2 blocks of padded convolution followed by batch_normalization,
    (thus preserving the dimentionality)
    '''
    def __init__(self, num_channels, num_filters):
        '''
        Init function, defines the layer variables
        '''
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_filters, (3, 3), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, (3, 3), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters)

    def forward(self, inp_tensor):
        '''
        Forward funtion, passes the data through the layers,
        applies batch normalization, and relu activation
        '''
        encoder = self.conv1(inp_tensor)
        encoder = self.conv1_bn(encoder)
        encoder = torch.relu(encoder)
        encoder = self.conv2(encoder)
        encoder = self.conv2_bn(encoder)
        encoder = torch.relu(encoder)
        return encoder

class encoder_block(nn.Module):
    '''
    The encoder module (for dimensionality reduction)
    '''
    def __init__(self, num_channels, num_filters):
        '''
        Init function, defines the layer variables
        '''
        super(encoder_block, self).__init__()
        self.conv_block1 = conv_block(num_channels, num_filters)
        self.max_pool1 = nn.MaxPool2d((2, 2), (2, 2))

    def forward(self, inp_tensor):
        '''
        Forward funtion, passes the data through the layers, and max pools
        '''
        encoder = self.conv_block1(inp_tensor)
        encoder_pool = self.max_pool1(encoder)
        return (encoder_pool, encoder)

class decoder_block(nn.Module):
    '''
    The decoder module (for dimensionality restoration)
    '''
    def __init__(self, num_channels, num_filters):
        '''
        Init function, defines the layer variables
        '''
        super(decoder_block, self).__init__()
        self.conv_tp1 = nn.ConvTranspose2d(num_channels, num_filters,\
                                           (2, 2), stride=(2, 2))
        self.conv_tp1_bn = nn.BatchNorm2d(2 * num_filters)
        self.conv_tp2 = nn.Conv2d(2 * num_filters, num_filters,
                                  (3, 3), padding=1)
        self.conv_tp2_bn = nn.BatchNorm2d(num_filters)
        self.conv_tp3 = nn.Conv2d(num_filters, num_filters, (3, 3), padding=1)
        self.conv_tp3_bn = nn.BatchNorm2d(num_filters)

    def forward(self, inp_tensor, concat_tensor):
        '''
        Forward funtion, accepts previous layer outputs
        along with encoder output at the same level,
        concatenates them, and passes them through the layers.
        Finally after each layer, performs batch normalization,
        and relu activation.
        '''
        decoder = self.conv_tp1(inp_tensor)
        decoder = torch.cat((concat_tensor, decoder), 1)
        decoder = self.conv_tp1_bn(decoder)
        decoder = torch.relu(decoder)
        decoder = self.conv_tp2(decoder)
        decoder = self.conv_tp2_bn(decoder)
        decoder = torch.relu(decoder)
        decoder = self.conv_tp3(decoder)
        decoder = self.conv_tp3_bn(decoder)
        decoder = torch.relu(decoder)
        return decoder


class UNet2D(nn.Module):
    '''
    UNet model that accepts 2D input slices, and outputs segmentations
    '''
    def __init__(self, num_channels=1):
        '''
        accepts number of output channels
        and defines the encoder/decoder block variables
        '''
        super(UNet2D, self).__init__()
        self.num_channels = num_channels
        self.encoder_block0 = encoder_block(num_channels, 32)
        self.encoder_block1 = encoder_block(32, 64)
        self.encoder_block2 = encoder_block(64, 128)
        self.encoder_block3 = encoder_block(128, 256)
        self.encoder_block4 = encoder_block(256, 512)
        self.center = conv_block(512, 1024)
        self.decoder_block4 = decoder_block(1024, 512)
        self.decoder_block3 = decoder_block(512, 256)
        self.decoder_block2 = decoder_block(256, 128)
        self.decoder_block1 = decoder_block(128, 64)
        self.decoder_block0 = decoder_block(64, 32)
        self.conv_final = nn.Conv2d(32, num_channels, (1, 1))

    def forward(self, inputs):
        '''
        forward function to pass the data correctly
        the dimension of the filters after each layer has been mentioned
        along with the number of channels
        (since square input, hence one side dimension mentioned)
        '''
        # inputs = x # 256 (c=1)
        # Encoder section
        encoder0_pool, encoder0 = self.encoder_block0(inputs) # 128 (c=32)
        encoder1_pool, encoder1 = self.encoder_block1(encoder0_pool) # 64 (c=64)
        encoder2_pool, encoder2 = self.encoder_block2(encoder1_pool) # 32 (c=128)
        encoder3_pool, encoder3 = self.encoder_block3(encoder2_pool) # 16 (c=256)
        encoder4_pool, encoder4 = self.encoder_block4(encoder3_pool) # 8 (c=512)
        # Feature block
        center = self.center(encoder4_pool) # center (dim = 8, c=1024)
        # Decoder section
        decoder4 = self.decoder_block4(center, encoder4) # 16 (c=512)
        decoder3 = self.decoder_block3(decoder4, encoder3) # 32 (c=256)
        decoder2 = self.decoder_block2(decoder3, encoder2) # 64 (c=129)
        decoder1 = self.decoder_block1(decoder2, encoder1) # 128 (c=64)
        decoder0 = self.decoder_block0(decoder1, encoder0) # 256 (c=32)
        # Output layer with activation
        if self.num_channels == 1:
            outputs = torch.sigmoid(self.conv_final(decoder0)) # (c=1)
        else:
            outputs = torch.softmax(self.conv_final(decoder0), axis=1) # (c=n)
        return outputs

#######################################################
# Separated encoder and decoder part of UNet for MONet
# out for deocder duplication purpose while fine-tuning
#######################################################

class MO_Net_encoder(nn.Module):
    '''
    MO-Net encoder section
    (same as the encoder part of UNet)
    '''
    def __init__(self):
        '''
        Defining encoder blocks of UNet
        '''
        super(MO_Net_encoder, self).__init__()
        self.encoder_block0 = encoder_block(num_channels, 32)
        self.encoder_block1 = encoder_block(32, 64)
        self.encoder_block2 = encoder_block(64, 128)
        self.encoder_block3 = encoder_block(128, 256)
        self.encoder_block4 = encoder_block(256, 512)
        self.center = conv_block(512, 1024)

    def forward(self, inputs):
        '''
        Passing data through the encoder layers
        Channels and dimensions same as UNet layer (refer above for details)
        '''
        # inputs = x # 256
        encoder0_pool, encoder0 = self.encoder_block0(inputs) # 128
        encoder1_pool, encoder1 = self.encoder_block1(encoder0_pool) # 64
        encoder2_pool, encoder2 = self.encoder_block2(encoder1_pool) # 32
        encoder3_pool, encoder3 = self.encoder_block3(encoder2_pool) # 16
        encoder4_pool, encoder4 = self.encoder_block4(encoder3_pool) # 8
        center = self.center(encoder4_pool) # center (8)

        return center


class MO_Net_decoder(nn.Module):
    '''
    MO-Net decoder section
    (same as the decoder part of UNet)
    '''
    def __init__(self, num_channels=1):
        '''
        Defining decoder blocks of UNet
        '''
        super(MO_Net_decoder, self).__init__()
        self.num_channels = num_channels
        self.decoder_block4 = decoder_block(1024, 512)
        self.decoder_block3 = decoder_block(512, 256)
        self.decoder_block2 = decoder_block(256, 128)
        self.decoder_block1 = decoder_block(128, 64)
        self.decoder_block0 = decoder_block(64, 32)
        self.conv_final = nn.Conv2d(32, num_channels, (1, 1))

    def forward(self, center):
        '''
        Passing data through the decoder layer to generate segmentations
        '''
        # center -> 8
        decoder4 = self.decoder_block4(center, encoder4) # 16
        decoder3 = self.decoder_block3(decoder4, encoder3) # 32
        decoder2 = self.decoder_block2(decoder3, encoder2) # 64
        decoder1 = self.decoder_block1(decoder2, encoder1) # 128
        decoder0 = self.decoder_block0(decoder1, encoder0) # 256

        if self.num_channels == 1:
            outputs = torch.sigmoid(self.conv_final(decoder0))
        else:
            outputs = torch.softmax(self.conv_final(decoder0), axis=1)
        return outputs
