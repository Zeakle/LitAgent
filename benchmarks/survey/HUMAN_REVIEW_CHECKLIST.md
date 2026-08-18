# Survey Benchmark Human Review Checklist

## Purpose

This checklist records the source audit and remains the upgrade path to independent human review. The
source audit performed on 2026-08-13 verified identifiers, titles, query scope, paper relevance, and topic-to-paper links against
the source-linked arXiv metadata and abstracts. That audit can find inconsistencies, but it cannot truthfully certify the
dataset as `human_reviewed` on behalf of the project owner. The owner accepted the AI-assisted audit on
2026-08-13, so the current status is `owner_approved_ai_assisted`.

Do not tick a case after checking only its title. Read at least the abstract and method/contribution
summary for every paper, then decide whether the paper is required, relevant but optional, or a hard
negative for the exact query.

## Corrections Already Applied

- `few-shot-optimization`: renamed the MetaOptNet topic from vague "optimization stability" to its
  actual differentiable convex base-learner contribution.
- `few-shot-simple-baselines`: moved Distribution Calibration here because it augments pretrained
  features and a simple classifier; it is not a cross-domain or transductive method.
- `few-shot-vlm`: replaced Visual Prompt Tuning with MaPLe. VPT adapts vision transformers without a
  vision-language model, so it is now a deliberately close few-shot distractor.
- `nerf-foundations`: removed the request to compare earlier neural volumes because Neural Volumes is
  intentionally a distractor and was absent from the relevant judgment.
- Few-shot distractors: replaced Meta-Transfer Learning, which could answer optimization/simple-baseline
  queries, with the object-detection-specific DeFRCN.

## Sign-off Rules

For each case, verify all five boxes. `Required` means omission should materially fail the case;
`Relevant` means the paper can legitimately support the query but need not appear in every good answer.
Topic wording must describe a contribution actually supported by its `source_paper_ids`.

### Few-shot Image Classification

#### `few-shot-metric`

- Query: metric-learning approaches for few-shot image classification.
- Required: [Matching Networks](https://arxiv.org/abs/1606.04080),
  [Prototypical Networks](https://arxiv.org/abs/1703.05175).
- Relevant: [Relation Network](https://arxiv.org/abs/1711.06025),
  [TADAM](https://arxiv.org/abs/1805.10123),
  [DeepEMD](https://arxiv.org/abs/2003.06777).
- Topics: episodic matching, prototypes, learned/task-adaptive metrics, set-level EMD.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `few-shot-optimization`

- Query: optimization-based meta-learning for few-shot visual recognition.
- Required: [MAML](https://arxiv.org/abs/1703.03400),
  [Meta-SGD](https://arxiv.org/abs/1707.09835).
- Relevant: [MetaOptNet](https://arxiv.org/abs/1904.03758).
- Topics: learned initialization, learned update direction/rate, differentiable convex base learners.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `few-shot-simple-baselines`

- Query: competitiveness of simple transfer-learning baselines.
- Required: [A Closer Look](https://arxiv.org/abs/1904.04232),
  [A Good Embedding Is All You Need?](https://arxiv.org/abs/2003.11539).
- Relevant: [SimpleShot](https://arxiv.org/abs/1911.04623),
  [Meta-Baseline](https://arxiv.org/abs/2003.04390),
  [Distribution Calibration](https://arxiv.org/abs/2101.06395).
- Topics: pretrained-feature baseline strength, transfer versus meta-learning, feature-distribution
  calibration and augmentation.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `few-shot-cross-domain`

- Query: cross-domain and transductive few-shot image classification.
- Required: [Meta-Dataset](https://arxiv.org/abs/1903.03096),
  [TIM](https://arxiv.org/abs/2008.11297).
- Relevant: [CrossTransformers](https://arxiv.org/abs/2007.11498),
  [Unified Transfer/Meta Benchmark](https://arxiv.org/abs/2104.02638).
- Topics: transductive query-set inference, domain shift, practical transfer/meta-learning behavior.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `few-shot-vlm`

- Query: vision-language model adaptation for few-shot image recognition.
- Required: [CoOp](https://arxiv.org/abs/2109.01134),
  [CoCoOp](https://arxiv.org/abs/2203.05557).
- Relevant: [CLIP-Adapter](https://arxiv.org/abs/2110.04544),
  [Tip-Adapter](https://arxiv.org/abs/2111.03930),
  [MaPLe](https://arxiv.org/abs/2210.03117).
- Topics: prompt learning, parameter-efficient CLIP adaptation, multimodal prompt generalization.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

### Vision Transformers

#### `vit-foundations`

- Required: [ViT](https://arxiv.org/abs/2010.11929),
  [ResNet](https://arxiv.org/abs/1512.03385).
- Relevant: [ConvNeXt](https://arxiv.org/abs/2201.03545).
- Topics: patch tokens/global attention and the convolutional inductive-bias comparison.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `vit-data-efficient`

- Required: [DeiT](https://arxiv.org/abs/2012.12877).
- Relevant: [Tokens-to-Token ViT](https://arxiv.org/abs/2101.11986).
- Topics: distillation token and progressive token aggregation for data-efficient training.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `vit-hierarchical`

- Required: [Swin Transformer](https://arxiv.org/abs/2103.14030),
  [Pyramid Vision Transformer](https://arxiv.org/abs/2102.12122).
- Relevant: [CvT](https://arxiv.org/abs/2103.15808).
- Topics: shifted windows, feature pyramids, convolutional projections and hierarchical tokens.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `vit-self-supervised`

- Required/relevant: [DINO](https://arxiv.org/abs/2104.14294),
  [MAE](https://arxiv.org/abs/2111.06377).
- Topics: momentum-teacher self-distillation and high-ratio masked reconstruction.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `vit-efficiency`

- Required: [LeViT](https://arxiv.org/abs/2104.01136),
  [MobileViT](https://arxiv.org/abs/2110.02178).
- Relevant: [EfficientFormer](https://arxiv.org/abs/2206.01191).
- Topics: hybrid efficient designs, resolution schedules, hardware-aware measured latency.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

### Neural Radiance Fields

#### `nerf-foundations`

- Required/relevant: [NeRF](https://arxiv.org/abs/2003.08934).
- Topics: continuous density/view-dependent radiance and differentiable ray integration.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `nerf-acceleration`

- Required: [Instant-NGP](https://arxiv.org/abs/2201.05989),
  [Plenoxels](https://arxiv.org/abs/2112.05131).
- Relevant: [DVGO](https://arxiv.org/abs/2111.11215).
- Topics: multiresolution hash encoding and explicit sparse/direct voxel grids.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `nerf-sparse-generalization`

- Required: [pixelNeRF](https://arxiv.org/abs/2012.02190),
  [IBRNet](https://arxiv.org/abs/2102.13090).
- Relevant: [MVSNeRF](https://arxiv.org/abs/2103.15595).
- Topics: image-conditioned fields and multi-view feature aggregation/cross-scene generalization.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `nerf-antialias-unbounded`

- Required: [Mip-NeRF](https://arxiv.org/abs/2103.13415),
  [Mip-NeRF 360](https://arxiv.org/abs/2111.12077).
- Relevant: [NeRF++](https://arxiv.org/abs/2010.07492).
- Topics: integrated conical frustums and unbounded-scene parameterization.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

#### `nerf-dynamic`

- Required: [D-NeRF](https://arxiv.org/abs/2011.13961),
  [Nerfies](https://arxiv.org/abs/2011.12948).
- Relevant: [HyperNeRF](https://arxiv.org/abs/2106.13228).
- Topics: canonical-space deformation and higher-dimensional topology changes.
- [ ] Query scope [ ] Required set [ ] Relevant set [ ] Topic links [ ] Approve case

## Corpus Distractors

Confirm that these are plausible retrieval candidates but irrelevant to the domain's five queries:

- Few-shot: [DeFRCN](https://arxiv.org/abs/2108.09017) is an object-detection method, and
  [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119) adapts vision-only transformers rather than
  vision-language models.
- Vision Transformer: [MLP-Mixer](https://arxiv.org/abs/2105.01601),
  [ResMLP](https://arxiv.org/abs/2105.03404).
- NeRF: [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079),
  [Neural Volumes](https://arxiv.org/abs/1906.07751).

- [ ] All six distractors are intentionally irrelevant to every case in their domain.
- [ ] The corpus equals the deduplicated relevant-paper union plus these six distractors.

## Owner Acceptance And Optional Human Sign-off

- AI source auditor: Codex
- Owner acceptance date: 2026-08-13
- Owner decision: accepted for the project baseline
- Dataset version reviewed: `1.1.0`
- [x] The owner accepts the disclosed AI-assisted audit and its five corrections.
- [x] `review_notes` and `annotation_version` preserve the audit provenance.
- [ ] An independent human has checked all 15 cases and six distractors paper by paper.
- [ ] After independent review, `judgment_status` may be upgraded to `human_reviewed`.
