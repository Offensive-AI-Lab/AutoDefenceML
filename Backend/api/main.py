import torch
def check_cuda():
    if torch.cuda.is_available():
        print("Yes")
    else:
        print("No",torch.cuda.get_device_name(torch.cuda.current_device()))

if __name__ == '__main__':
    check_cuda()