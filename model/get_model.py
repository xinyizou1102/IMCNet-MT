from model.MotionSeg import MotionSOD


def get_model(option):
    model = MotionSOD(option).cuda()
    print("[INFO]: Generator have {:.4f}Mb paramerters in total".format(sum(x.numel()/1e6 for x in model.parameters())))

    return model
