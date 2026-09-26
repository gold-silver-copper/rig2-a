//! Vision tasks: classification, segmentation, detection and pose estimation.
//!
//! Inputs are encoded images (PNG, JPEG, ...) with their media type; decoding
//! and preprocessing belong to the model. All coordinates are in pixels of
//! the input image, with the origin at the top-left corner, x to the right
//! and y down.

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{StreamingTask, Task};

/// An encoded image: its bytes and media type.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EncodedImage {
    /// The media type, such as `image/png`.
    pub media_type: String,
    /// The encoded bytes, serialized as base64.
    #[serde(with = "crate::base64_bytes")]
    pub bytes: Bytes,
}

impl EncodedImage {
    /// A PNG image.
    pub fn png(bytes: impl Into<Bytes>) -> Self {
        Self {
            media_type: "image/png".into(),
            bytes: bytes.into(),
        }
    }

    /// A JPEG image.
    pub fn jpeg(bytes: impl Into<Bytes>) -> Self {
        Self {
            media_type: "image/jpeg".into(),
            bytes: bytes.into(),
        }
    }
}

/// Label an image.
#[derive(Debug, Clone, Copy)]
pub struct ImageClassification;

impl Task for ImageClassification {
    const NAME: &'static str = "image_classification";
    type Input = ClassificationRequest;
    type Output = Vec<Label>;
    type Capabilities = VisionCapabilities;
}

/// An image to classify.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationRequest {
    /// The image.
    pub image: EncodedImage,
    /// How many labels to return, best first.
    pub top_k: u32,
}

impl From<EncodedImage> for ClassificationRequest {
    fn from(image: EncodedImage) -> Self {
        Self { image, top_k: 5 }
    }
}

/// A class and its probability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label {
    /// The class index in the model's label set.
    pub class_id: u32,
    /// The class name.
    pub label: String,
    /// The probability, from 0 to 1.
    pub score: f32,
}

/// Label every pixel of an image.
#[derive(Debug, Clone, Copy)]
pub struct SemanticSegmentation;

impl Task for SemanticSegmentation {
    const NAME: &'static str = "semantic_segmentation";
    type Input = EncodedImage;
    type Output = SegmentationMask;
    type Capabilities = VisionCapabilities;
}

/// A class index for every pixel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationMask {
    /// Width in pixels, equal to the input's.
    pub width: u32,
    /// Height in pixels, equal to the input's.
    pub height: u32,
    /// The class names, indexed by class id.
    pub classes: Vec<String>,
    /// One class id per pixel, row by row from the top-left.
    pub mask: Vec<u16>,
}

impl SegmentationMask {
    /// The class id at pixel `(x, y)`, if it is inside the mask.
    pub fn class_at(&self, x: u32, y: u32) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.mask.get((y * self.width + x) as usize).copied()
    }
}

/// Find objects in an image.
#[derive(Debug, Clone, Copy)]
pub struct ObjectDetection;

impl Task for ObjectDetection {
    const NAME: &'static str = "object_detection";
    type Input = DetectionRequest;
    type Output = Vec<Detection>;
    type Capabilities = VisionCapabilities;
}

/// An image to search, and the thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionRequest {
    /// The image.
    pub image: EncodedImage,
    /// Drop detections scored below this, from 0 to 1.
    pub min_score: f32,
    /// Suppress overlapping boxes with an intersection-over-union above this.
    pub max_overlap: f32,
}

impl From<EncodedImage> for DetectionRequest {
    fn from(image: EncodedImage) -> Self {
        Self {
            image,
            min_score: 0.25,
            max_overlap: 0.45,
        }
    }
}

/// An axis-aligned box in input pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Left edge.
    pub x_min: f32,
    /// Top edge.
    pub y_min: f32,
    /// Right edge.
    pub x_max: f32,
    /// Bottom edge.
    pub y_max: f32,
}

impl BoundingBox {
    /// The area; zero for an inverted box.
    pub fn area(&self) -> f32 {
        (self.x_max - self.x_min).max(0.0) * (self.y_max - self.y_min).max(0.0)
    }

    /// Intersection over union with `other`.
    pub fn iou(&self, other: &Self) -> f32 {
        let overlap = Self {
            x_min: self.x_min.max(other.x_min),
            y_min: self.y_min.max(other.y_min),
            x_max: self.x_max.min(other.x_max),
            y_max: self.y_max.min(other.y_max),
        }
        .area();
        let union = self.area() + other.area() - overlap;
        if union <= 0.0 { 0.0 } else { overlap / union }
    }
}

/// One found object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// The class index in the model's label set.
    pub class_id: u32,
    /// The class name.
    pub label: String,
    /// The confidence, from 0 to 1.
    pub score: f32,
    /// Where it is.
    pub bbox: BoundingBox,
}

/// Find people and their keypoints in a sequence of frames.
///
/// It streams one [`FramePoses`] per frame, so a video can be processed as
/// it decodes.
#[derive(Debug, Clone, Copy)]
pub struct PoseEstimation;

impl Task for PoseEstimation {
    const NAME: &'static str = "pose_estimation";
    type Input = PoseRequest;
    type Output = Vec<FramePoses>;
    type Capabilities = VisionCapabilities;
}

impl StreamingTask for PoseEstimation {
    type Event = FramePoses;
}

/// Frames to search, and the thresholds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseRequest {
    /// The frames, in order.
    pub frames: Vec<EncodedImage>,
    /// Drop people scored below this, from 0 to 1.
    pub min_score: f32,
    /// Suppress overlapping people with an intersection-over-union above this.
    pub max_overlap: f32,
}

impl From<EncodedImage> for PoseRequest {
    fn from(frame: EncodedImage) -> Self {
        Self {
            frames: vec![frame],
            min_score: 0.25,
            max_overlap: 0.45,
        }
    }
}

impl From<Vec<EncodedImage>> for PoseRequest {
    fn from(frames: Vec<EncodedImage>) -> Self {
        Self {
            frames,
            min_score: 0.25,
            max_overlap: 0.45,
        }
    }
}

/// The people found in one frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FramePoses {
    /// The frame's position in the request.
    pub frame: u32,
    /// The people found.
    pub poses: Vec<Pose>,
}

/// One person: a box, a score and keypoints in [`Skeleton::COCO`] order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    /// The confidence that this is a person, from 0 to 1.
    pub score: f32,
    /// Where the person is.
    pub bbox: BoundingBox,
    /// The keypoints, in the skeleton's order.
    pub keypoints: Vec<Keypoint>,
}

/// A body keypoint in input pixels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Keypoint {
    /// Horizontal position.
    pub x: f32,
    /// Vertical position.
    pub y: f32,
    /// Visibility confidence, from 0 to 1.
    pub score: f32,
}

/// A keypoint layout: the name of each keypoint and the limbs joining them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Skeleton {
    /// Keypoint names, by index.
    pub keypoints: &'static [&'static str],
    /// Pairs of keypoint indices joined by a limb.
    pub limbs: &'static [(u8, u8)],
}

impl Skeleton {
    /// The 17-keypoint COCO layout.
    pub const COCO: Self = Self {
        keypoints: &[
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        limbs: &[
            (15, 13),
            (13, 11),
            (16, 14),
            (14, 12),
            (11, 12),
            (5, 11),
            (6, 12),
            (5, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (1, 2),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
        ],
    };
}

/// What a vision model declares: its input size and label set.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct VisionCapabilities {
    /// The width and height the model resizes inputs to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_size: Option<(u32, u32)>,
    /// The class names, by class id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}
