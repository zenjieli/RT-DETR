import os.path as osp
import PIL.Image as Image

def main(args):
    model = RTDETR(args.checkpoint)
    results = model(osp.expanduser('~/Pictures/frame-065.jpg'))

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()
    return args

"""
self.ema.load_state_dict(state['ema'])
"""

if __name__ == '__main__':
    args = parse_args()
    main()