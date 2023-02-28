import torch



if __name__ == "__main__":
    from mmdet.apis.inference import init_detector
    from custom_tools.vis import draw_instance_seg_results
    import cv2
    from torchvision.models import resnet18

    device = "cpu"


    model = resnet18(pretrained=False)
    torch.save(model, "x.pth")
    exit()
    model.to(device).eval()

    # model_script = torch.jit.trace(model, example_inputs=(torch.rand((1, 3, 224, 224), dtype=torch.float32), ))
    model_script = torch.jit.script(model)
    model_script.to(device).eval()

    # save module
    model_script.save("resnet18_cpu.pt")
    exit()

    # # load module
    # model_script = torch.jit.load(jit_module_save_path).to(device).eval()

