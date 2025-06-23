import warnings

warnings.filterwarnings("ignore")

import os, sys, json, datetime

sys.path.append("./submodules/VAR")
sys.path.append("./submodules/VAR/models")

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from PIL import Image
from tqdm.auto import tqdm

import torch
from torchvision import transforms

import tensorflow.compat.v1 as tf  # type: ignore

from src import (
    get_path,
    Evaluator,
    eval_models,
    set_randomness,
    create_npz_from_sample_folder,
)

from submodules.VAR.models.var import VAR


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    model_cfg = cfg.model
    finetuning_cfg = cfg.finetuning
    trainer_cfg = cfg.trainer
    action_cfg = cfg.action
    dataset_cfg = cfg.dataset
    config = cfg.cfg
    with open_dict(model_cfg):
        model_cfg.device = trainer_cfg.device
        model_cfg.seed = config.seed

    action_input = [
        config,
        model_cfg,
        finetuning_cfg,
        trainer_cfg,
        action_cfg,
        dataset_cfg,
    ]
    for cf in action_input:
        print(OmegaConf.to_yaml(cf))

    if action_cfg.name == "scores_computation":
        scores_computation(*action_input)
    elif action_cfg.name == "generate_samples":
        generate_samples(*action_input)
    else:
        raise ValueError("Invalid action name")

    print("fin")


def scores_computation(
    config: DictConfig,
    model_cfg: DictConfig,
    finetuning_cfg: DictConfig,
    trainer_cfg: DictConfig,
    action_cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> None:
    """
    >>> 1. Compile .npz file for generated samples
    >>> 2. Compile .npz file for reference samples
    >>> 3. Compute FID, sFID, IS, Precision and Recall
    """

    gen_images_path = get_path(
        x=config.generated_sample_path,
        config=config,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        finetuning_cfg=finetuning_cfg,
        dataset_cfg=dataset_cfg,
    )
    ref_images_path = os.path.join(
        config.reference_sample_path,
        dataset_cfg.name,
        dataset_cfg.split,
    )

    # Compile Generated Samples
    npz_gen_path = os.path.join(
        config.path_to_scores,
        get_path(
            x="",
            config=config,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            finetuning_cfg=finetuning_cfg,
            dataset_cfg=dataset_cfg,
        ).replace("/", "_"),
    )
    os.makedirs(config.path_to_scores, exist_ok=True)
    print("Compiling generated samples into .npz file...")
    gen_npz_file = create_npz_from_sample_folder(
        path_to_save=npz_gen_path,
        sample_folder=gen_images_path,
        type="gen",
    )
    gen_npz_file = os.path.join(npz_gen_path, "generated_samples.npz")
    print(f"Generated samples compiled into .npz file: {gen_npz_file}")

    # Compile Reference Samples
    npz_ref_path = os.path.join(
        config.path_to_scores,
        dataset_cfg.name,
        dataset_cfg.split,
    )
    os.makedirs(config.path_to_scores, exist_ok=True)

    # Check if reference NPZ file already exists
    ref_npz_file = os.path.join(npz_ref_path, "reference_samples.npz")
    if os.path.exists(ref_npz_file):
        print(f"Reference NPZ file already exists at: {ref_npz_file}")
    else:
        print("Compiling reference samples into .npz file...")
        ref_npz_file = create_npz_from_sample_folder(
            sample_folder=ref_images_path,
            path_to_save=npz_ref_path,
            type="ref",
        )
        print(f"Reference samples compiled into .npz file: {ref_npz_file}")

    print("Initializing TensorFlow evaluator...")
    config_tf = tf.ConfigProto(
        allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
    )
    config_tf.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config_tf))
    print("Warming up TensorFlow...")
    evaluator.warmup()

    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_npz_file)
    print("Computing reference batch statistics...")
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_npz_file, ref_acts)

    print("Computing sample batch activations...")
    sample_acts = evaluator.read_activations(gen_npz_file)
    print("Computing sample batch statistics...")
    sample_stats, sample_stats_spatial = evaluator.read_statistics(
        gen_npz_file, sample_acts
    )

    # Calculate and print metrics
    print("\nComputing evaluation metrics...")
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    precision, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    print("#" * 50)
    print("Evaluation Results:")
    print(f"FID: {fid:.3f}")
    print(f"sFID: {sfid:.3f}")
    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print("#" * 50)

    print("\nSaving metrics to JSON file...")
    metrics = {
        "Run_ID": str(
            get_path(
                x="",
                config=config,
                model_cfg=model_cfg,
                trainer_cfg=trainer_cfg,
                finetuning_cfg=finetuning_cfg,
                dataset_cfg=dataset_cfg,
            ).replace("/", "_")
        ),
        "DateTime": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "Model": str(model_cfg.name),
        "Trainer": str(trainer_cfg.name),
        "Finetuning": str(finetuning_cfg.name),
        "Dataset": str(dataset_cfg.name),
        "Split": str(dataset_cfg.split),
        "Checkpoint": str(trainer_cfg.checkpoint),
        "FID": float(fid),
        "sFID": float(sfid),
        "Precision": precision,
        "Recall": recall,
    }

    path_to_metrics = config.path_to_metrics
    os.makedirs(path_to_metrics, exist_ok=True)

    metrics_file = os.path.join(
        path_to_metrics,
        get_path(
            x="",
            config=config,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            finetuning_cfg=finetuning_cfg,
            dataset_cfg=dataset_cfg,
        ).replace("/", "_")
        + ".json",
    )
    if os.path.exists(metrics_file):
        print(f"\nMetrics file already exists. Deleting the old file...")
        os.remove(metrics_file)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_file}")


def generate_samples(
    config: DictConfig,
    model_cfg: DictConfig,
    finetuning_cfg: DictConfig,
    trainer_cfg: DictConfig,
    action_cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> None:
    """
    >>> 1. Loads the finetuned model
    >>> 2. Generates and saves the samples, where for each class the number of generated samples
        equals the number of images found in the test set for that class.
    >>> 3. Selects and saves all the reference samples (Real Images) from the test set.
    """

    set_randomness(config.seed)

    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")

    dtype_dict = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_dict.get(action_cfg.gen_params.dtype, None)
    if dtype is None:
        raise ValueError(f"Invalid dtype: {action_cfg.gen_params.dtype}")

    def transform_image_for_reference(image_path: str, image_size: tuple = (256, 256)):
        """
        Preprocesses an image from `image_path` for FID evaluation by combining
        center cropping (to the shortest side) and resizing to (256,256).
        If image size already matches target size, returns original image.
        """
        img = Image.open(image_path).convert("RGB")

        # If image dimensions match target size, return original
        if img.size == image_size:
            return img

        transform = transforms.Compose(
            [transforms.CenterCrop(min(img.size)), transforms.Resize(image_size)]
        )
        return transform(img)

    # Initializtion
    model: VAR = eval_models[model_cfg.name][finetuning_cfg.name](
        config=config,
        model_cfg=model_cfg,
        finetuning_cfg=finetuning_cfg,
        trainer_cfg=trainer_cfg,
        dataset_cfg=dataset_cfg,
        action_cfg=action_cfg,
    )

    num_classes = dataset_cfg.num_classes
    print(
        f"[INFO] Number of classes: {num_classes} | Generating samples based on the {dataset_cfg.split} set sample counts per class"
    )

    ######## Generated Samples ########
    gen_images_path = get_path(
        x=config.generated_sample_path,
        config=config,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        finetuning_cfg=finetuning_cfg,
        dataset_cfg=dataset_cfg,
    )
    os.makedirs(gen_images_path, exist_ok=True)
    print(f"[{action_cfg.name}] Saving Generated samples to: {gen_images_path}")

    batch_size = 16
    all_img_tensors = []  # List to store tensors for all classes
    test_images_path = os.path.join(dataset_cfg.dataset_path, dataset_cfg.split)

    for idx in tqdm(range(0, num_classes), desc="Generating images for each class"):
        # Get test images for this class
        class_test_path = os.path.join(test_images_path, str(idx))
        image_files = sorted(
            [f for f in os.listdir(class_test_path) if f.lower().endswith(".png")]
        )
        total_samples = len(image_files)
        print(f"Generating {total_samples} samples for class {idx}")
        if total_samples == 0:
            raise ValueError(
                f"No images found in test set for class {idx} at {class_test_path}"
            )

        class_tensor_list = []

        for start in range(0, total_samples, batch_size):

            current_batch_size = min(batch_size, total_samples - start)
            class_labels = [idx] * current_batch_size
            B = len(class_labels)
            label_B = torch.tensor(class_labels, device=action_cfg.device)

            with torch.inference_mode():
                with torch.autocast(
                    "cuda", enabled=True, dtype=dtype, cache_enabled=True
                ):
                    recon_B3HW = model.var.autoregressive_infer_cfg(
                        B=B,
                        label_B=label_B,
                        cfg=model_cfg.sampling_params.cfg,
                        top_k=model_cfg.sampling_params.top_k,
                        top_p=model_cfg.sampling_params.top_p,
                        g_seed=config.seed,
                        more_smooth=model_cfg.sampling_params.more_smooth,
                    )

                if dtype == torch.bfloat16:
                    recon_B3HW = recon_B3HW.float()

            class_tensor_list.append(recon_B3HW.cpu())

        class_tensor_cat = torch.cat(class_tensor_list, dim=0)
        all_img_tensors.append(class_tensor_cat)

    # Combine all tensors and convert them to images.
    all_img_tensors = torch.cat(all_img_tensors, dim=0)
    generated_images = []
    for img_tensor in tqdm(all_img_tensors, desc="Converting tensors to images"):
        img_pil = Image.fromarray(
            (img_tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        generated_images.append(img_pil)

    # Save the generated images.
    global_counter = 1
    for img in tqdm(generated_images, desc="Saving generated images"):
        img_path = os.path.join(gen_images_path, f"{global_counter}.png")
        img.save(img_path)
        global_counter += 1
    print(f"Generated samples saved to: {gen_images_path}")

    ######## Reference Samples ########
    ref_images_path = os.path.join(dataset_cfg.dataset_path, dataset_cfg.split)
    ref_save_path = os.path.join(
        config.reference_sample_path,
        dataset_cfg.name,
        dataset_cfg.split,
    )
    os.makedirs(ref_save_path, exist_ok=True)

    # Compute total number of reference images across all classes.
    total_ref_samples = 0
    for idx in range(num_classes):
        class_path = os.path.join(ref_images_path, str(idx))
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
        total_ref_samples += len(image_files)

    existing_files = [
        f for f in os.listdir(ref_save_path) if f.lower().endswith(".png")
    ]
    if len(existing_files) >= total_ref_samples:
        print(
            f"[{action_cfg.name}] Reference samples already exist in: {ref_save_path}"
        )
    else:
        print(f"[{action_cfg.name}] Saving Reference samples to: {ref_save_path}")
        global_counter = 1
        for idx in tqdm(
            range(num_classes), desc="Selecting reference samples for each class"
        ):
            class_path = os.path.join(ref_images_path, str(idx))
            image_files = sorted(
                [f for f in os.listdir(class_path) if f.lower().endswith(".png")]
            )
            for img_name in image_files:
                src_img_path = os.path.join(class_path, img_name)
                transformed_img = transform_image_for_reference(
                    src_img_path, (model_cfg.image_size, model_cfg.image_size)
                )
                dest_img_path = os.path.join(ref_save_path, f"{global_counter}.png")
                transformed_img.save(dest_img_path)
                global_counter += 1
        print(f"Reference samples saved to: {ref_save_path}")

    # Check if count of generated samples matches the count of reference samples
    if global_counter - 1 != total_ref_samples:
        raise ValueError(
            f"Count mismatch: Generated samples: {global_counter - 1} | Reference samples: {total_ref_samples}"
        )


if __name__ == "__main__":
    main()
