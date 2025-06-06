import os
import argparse
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--imgs_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--delete_imgs", action="store_true", default=False)
    parser.add_argument("--video_name", type=str, default="video")
    args = parser.parse_args()

    imgs_dir = args.imgs_dir
    fps = args.fps
    delete_imgs = args.delete_imgs

    fnames = [f for f in os.listdir(imgs_dir) if f.lower().endswith('.png')]
    fnames = sorted(fnames)

    print(f"{os.path.join(imgs_dir, fnames[0])} -> {os.path.join(imgs_dir, fnames[-1])}")
    
    imgs = [cv2.imread(os.path.join(imgs_dir, fn), cv2.IMREAD_COLOR) for fn in fnames]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    size = imgs[0].shape[:2]
    videoWriter = cv2.VideoWriter(os.path.join(imgs_dir, f"{args.video_name}.mp4"), fourcc, fps, (size[1], size[0]))

    for i in range(len(imgs)):
        # cv2.imshow("test", imgs[i])
        # cv2.waitKey(0)
        videoWriter.write(imgs[i])
    
    videoWriter.release()

    print("video saved at {}".format(os.path.join(imgs_dir, f"{args.video_name}.mp4")))

    if delete_imgs:
        for i in range(len(imgs)):
            os.remove(os.path.join(imgs_dir, fnames[i]))
