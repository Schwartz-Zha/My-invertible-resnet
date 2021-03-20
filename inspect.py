import torch

def main():
    PATH = '/home/billy/PycharmProjects/bak/My-invertible-resnet/results/test_attention_concat_NoNewGamma/ims'
    #PATH = '/home/billy/PycharmProjects/bak/My-invertible-resnet/results/test_attention_concat_convGamma/ims'

    input_before = torch.load(PATH + '/input_before.pt')
    input_after = torch.load(PATH + '/input_after.pt')
    inverse_input_before = torch.load(PATH + '/inverse_input_before.pt')
    inverse_input_after = torch.load(PATH + '/inverse_input_after.pt')


    print('input_before')
    print(input_before[0,:,:,:])
    print('input_after')
    print(input_after[0,:,:,:])
    print(torch.min(input_after))
    print(torch.max(input_after))



if __name__ == '__main__':
    main()