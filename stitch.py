from matplotlib import pyplot as plt
from pathlib import Path
from stitching.image_handler import ImageHandler
import cv2 as cv
from stitching.feature_detector import FeatureDetector
import numpy as np
from stitching.feature_matcher import FeatureMatcher
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
import pickle
from pickle import dump, load
from tqdm import tqdm
import imutils

def get_image_paths(img_set):
    image_paths = [str(path.relative_to('.')) for path in Path('Frames').rglob(f'{img_set}*')]
    sorted_paths = sorted(image_paths)
    return sorted_paths

class Stitcher:
    def __init__(self):
		# determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, computeHomography=True, left=True):
        #for the first frame to stitch, compute the homography matrix and save it
        if computeHomography:
            basket_imgs = get_image_paths(images)
            img_handler = ImageHandler()
            img_handler.set_img_names(basket_imgs)

            #resize the images to medium (and later to low) resolution
            medium_imgs = list(img_handler.resize_to_medium_resolution())

            #find features
            finder = FeatureDetector(detector="orb", nfeatures=10000)
            features = [finder.detect_features(img) for img in medium_imgs]

            #match the features of the pairwise images
            matcher = FeatureMatcher()
            matches = matcher.match_features(features)

            #calibrate cameras which can be used to warp the images
            camera_estimator = CameraEstimator()
            camera_adjuster = CameraAdjuster()
            wave_corrector = WaveCorrector()

            cameras = camera_estimator.estimate(features, matches)
            cameras = camera_adjuster.adjust(features, matches, cameras)
            cameras = wave_corrector.correct(cameras)

            # Save camera calibration parameters
            self.saveCameraCalib(cameras)

            #warp the images into the final plane
            panorama =  self.warping_blending(img_handler, cameras)
        else:
            basket_imgs = get_image_paths(images)
            img_handler = ImageHandler()
            img_handler.set_img_names(basket_imgs)

            #Get camera calibration parameters
            cameras = self.getCameraCalib()
            panorama = self.warping_blending(img_handler,cameras)
        return panorama

    def saveCameraCalib(self, cameras):
        # Create a cv.FileStorage object for writing to a file
        file_storage = cv.FileStorage('cameras.yml', cv.FILE_STORAGE_WRITE)

        # Write each camera's parameters to the file storage
        for i, camera in enumerate(cameras):
            file_storage.write(f'camera{i+1}_focal', camera.focal)
            file_storage.write(f'camera{i+1}_aspect', camera.aspect)
            file_storage.write(f'camera{i+1}_ppx', camera.ppx)
            file_storage.write(f'camera{i+1}_ppy', camera.ppy)
            file_storage.write(f'camera{i+1}_R', np.asarray(camera.R))
            file_storage.write(f'camera{i+1}_t', np.asarray(camera.t))

        # Release the file storage
        file_storage.release()

    def getCameraCalib(self):
        # Create a cv.FileStorage object for reading the file
        file_storage = cv.FileStorage('cameras.yml', cv.FILE_STORAGE_READ)

        # Read each camera's parameters from the file storage and assign them to a new 'cameras' variable
        cameras = []
        for i in range(1, 3):  
            focal = file_storage.getNode(f'camera{i}_focal').real()
            aspect = file_storage.getNode(f'camera{i}_aspect').real()
            ppx = file_storage.getNode(f'camera{i}_ppx').real()
            ppy = file_storage.getNode(f'camera{i}_ppy').real()
            R = file_storage.getNode(f'camera{i}_R').mat()
            t = file_storage.getNode(f'camera{i}_t').mat()

            camera = cv.detail.CameraParams()
            camera.focal = focal
            camera.aspect = aspect
            camera.ppx = ppx
            camera.ppy = ppy
            camera.R = R
            camera.t = t

            cameras.append(camera)

        # Release the file storage
        file_storage.release()
        return cameras

    def warping_blending(self, img_handler, cameras):
        medium_imgs = list(img_handler.resize_to_medium_resolution())
        low_imgs = list(img_handler.resize_to_low_resolution(medium_imgs))
        final_imgs = list(img_handler.resize_to_final_resolution())

        #select the warper
        warper = Warper(warper_type="paniniA1.5B1")
        #set the the medium focal length of the cameras as scale
        warper.set_scale(cameras)

        #warp low resolution images
        low_sizes = img_handler.get_low_img_sizes()
        camera_aspect = img_handler.get_medium_to_low_ratio()

        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

        #Warp final resolution images
        final_sizes = img_handler.get_final_img_sizes()
        camera_aspect = img_handler.get_medium_to_final_ratio()

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
        final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

        #estimate the largest joint interior rectangle and crop the single images accordingly
        cropper = Cropper()
        mask = cropper.estimate_panorama_mask(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
        lir = cropper.estimate_largest_interior_rectangle(mask)
        low_corners = cropper.get_zero_center_corners(low_corners)
        rectangles = cropper.get_rectangles(low_corners, low_sizes)
        overlap = cropper.get_overlap(rectangles[1], lir)
        intersection = cropper.get_intersection(rectangles[1], overlap)

        cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        cropped_low_masks = list(cropper.crop_images(warped_low_masks))
        cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = img_handler.get_low_to_final_ratio()
        cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

        #Seam masks to find a transition line between images with the least amount of interference
        seam_finder = SeamFinder()
        seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
        seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

        #exposure error compensation
        compensator = ExposureErrorCompensator()
        compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)
        compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                            for idx, (img, mask, corner) 
                            in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]

        #blending
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()
        return panorama

#build a video from the stitched images
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []  
    for i in range(128):
        filename=pathIn + "frame_" + str(i)+".jpg"
        #reading each files
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv.VideoWriter(pathOut,cv.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def stitch_images(stitcher):
    for i in tqdm(range(128)):
        images = "frame"+str(i)+"_"
        output_path = 'Result/frame_'+str(i)+'.jpg'
        if(i==0):
            panorama = stitcher.stitch(images, computeHomography=True)
        else:
            panorama = stitcher.stitch(images, computeHomography=False)
        #save the stitched images in Result folder
        cv.imwrite(output_path, panorama)

if __name__ == '__main__':
    stitcher = Stitcher()
    stitch_images(stitcher)
    pathIn= 'Result/'
    pathOut = 'output_video.avi'
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)