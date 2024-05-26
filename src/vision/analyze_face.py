from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from loguru import logger
from matplotlib import (
    patches,
    pyplot as plt,
)
from msrest.authentication import CognitiveServicesCredentials

from src.settings import settings


def analyze_face():
    credentials = CognitiveServicesCredentials(settings.AZURE_AI_SUBSCRIPTION_KEY)
    client = FaceClient(settings.AZURE_AI_ENDPOINT, credentials)

    features = [
        FaceAttributeType.occlusion,
        FaceAttributeType.blur,
        FaceAttributeType.glasses,
    ]

    image_file = str(
        settings.BASE_DIR / "src" / "vision" / "face-images" / "people.jpg"
    )

    # Get faces
    with open(image_file, mode="rb") as image_data:
        detected_faces = client.face.detect_with_stream(
            image=image_data, return_face_attributes=features, return_face_id=False
        )

        if len(detected_faces) == 0:
            logger.error("No faces detected.")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        plt.axis("off")
        image = plt.imread(image_file)
        color = "g"
        face_count = 0

        for face in detected_faces:
            face_count += 1
            logger.info(f"\nFace number {face_count}")

            detected_attributes = face.face_attributes.as_dict()
            if "blur" in detected_attributes:
                logger.info(" - Blur:")
                for blur_name in detected_attributes["blur"]:
                    logger.info(
                        f"   - {blur_name}: {detected_attributes['blur'][blur_name]}"
                    )

            if "occlusion" in detected_attributes:
                logger.info(" - Occlusion:")
                for occlusion_name in detected_attributes["occlusion"]:
                    logger.info(
                        f"   - {occlusion_name}: {detected_attributes['occlusion'][occlusion_name]}"
                    )

            if "glasses" in detected_attributes:
                logger.info(" - Glasses:{}".format(detected_attributes["glasses"]))

            r = face.face_rectangle
            rect = patches.Rectangle(
                (r.left, r.top),
                r.width,
                r.height,
                linewidth=3,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            annotation = f"Face number {face_count}"
            ax.annotate(annotation, (r.left, r.top), backgroundcolor=color)

        plt.imshow(image)
        output_file = str(
            settings.BASE_DIR / "src" / "vision" / "outputs" / "detected_faces3.jpg"
        )
        fig.savefig(output_file)

        logger.info("\nResults saved in", output_file)


if __name__ == "__main__":
    analyze_face()
