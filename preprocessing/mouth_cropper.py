import cv2
import os


class MouthCropper:

    def crop_mouths(self, input_folder, output_folder):

        os.makedirs(output_folder, exist_ok=True)

        total = 0

        for image_name in sorted(os.listdir(input_folder)):

            image_path = os.path.join(input_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            h, w = image.shape[:2]

            # Approximate mouth region
            x = int(w * 0.25)
            y = int(h * 0.60)

            mouth = image[
                y:h,
                x:int(w * 0.75)
            ]

            cv2.imwrite(
                os.path.join(output_folder, image_name),
                mouth
            )

            total += 1

        return total