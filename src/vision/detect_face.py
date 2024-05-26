import asyncio

from azure.ai import vision
from loguru import logger
from matplotlib import (
    patches,
    pyplot as plt,
)

from src.settings import settings


async def detect_face():
    file = str(settings.BASE_DIR / "src" / "vision" / "images" / "people.jpg")

    client = vision.VisionServiceOptions(
        endpoint=settings.AZURE_AI_ENDPOINT,
        key=settings.AZURE_AI_SUBSCRIPTION_KEY,
    )
    analysis_options = vision.ImageAnalysisOptions()
    analysis_options.features = vision.ImageAnalysisFeature.PEOPLE

    img = vision.VisionSource(file)
    analyzer = vision.ImageAnalyzer(client, img, analysis_options)
    result = analyzer.analyze()

    if result.reason != vision.ImageAnalysisResultReason.ANALYZED:
        error_details = vision.ImageAnalysisErrorDetails.from_result(result)
        logger.error(
            f"Analysis failed: {error_details.reason}: {error_details.message}"
        )
        return

    if result.people is None:
        logger.info("No people detected")
        return

    img = plt.imread(file)
    fig = plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax = fig.add_subplot(111)
    ax.imshow(img)

    for person in result.people:
        if person.confidence > 0.5:
            rect = patches.Rectangle(
                (person.bounding_box.x, person.bounding_box.y),
                person.bounding_box.w,
                person.bounding_box.h,
                edgecolor="r",
                facecolor="none",
                linewidth=3,
            )
            ax.add_patch(rect)

    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(
        str(settings.BASE_DIR / "src" / "vision" / "outputs" / "detected_people2.png"),
    )


if __name__ == "__main__":
    asyncio.run(detect_face())
