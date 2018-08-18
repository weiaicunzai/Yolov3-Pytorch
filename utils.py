





def parse_cfg(cfg_path):
    """Takes a cfg file and then parse it

    Args:
        cfg_path: file path to yolov3.cfg 

    Returns:
        return a list of blocks, Each blocks
        describes the block in the neuron networks
    """

    blocks = []
    block = {}
    with open(cfg_path) as cfg:
        for line in cfg.readlines():
            line = line.strip()

            #remove comment
            if line.startswith('#'):
                continue

            #remove empty line
            if not line:
                continue

            if line.startswith('['):
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
            
                block['type'] = line[1 : -1].rstrip()

            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
            #print(line.strip())
        blocks.append(block)

    return blocks
    #for b in blocks:
    #    print(b)
#import cProfile
#
#cProfile.runctx("parse_cfg('cfg/yolov3.cfg')", globals(), None)
