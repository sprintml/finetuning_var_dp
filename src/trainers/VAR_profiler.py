import os, time, json
from typing import List

import torch
import torch.profiler

from src.utils import dist
from src.trainers.VAR_finetuning import (
    VARTrainer,
    FFTVARTrainer,
    LoraVARTrainer,
    LNTuningVARTrainer,
)


class ProfilingVARTrainer(VARTrainer):
    def train_step_with_profiling(self, inp_B3HW, label_B):
        """
        Runs one training step with profiler enabled.
        This comprises the full forward and backward pass.
        """
        B, V = label_B.shape[0], self.vae_local.vocab_size

        # Start profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as prof:
            # Forward pass
            gt_idx_Bl: List[torch.Tensor] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            with self.var_opt.amp_ctx:
                # Forward pass
                logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
                loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(
                    B, -1
                )
                lw = self.loss_weight
                loss = loss.mul(lw).sum(dim=-1).mean()

                # Backward pass
                self.var_opt.backward_clip_step(loss=loss, stepping=True)

        return prof

    def train(self) -> None:
        """Run profiling instead of regular training"""
        print("\nInitializing VAR profiling...")

        self.setup()

        # Get batch from dataloader
        start_setup_time = time.time()
        batch = next(iter(self.ld_train))
        inp, label = batch

        # Move tensors to device
        inp = inp.to(dist.get_device(), non_blocking=True)
        label = label.to(dist.get_device(), non_blocking=True)

        # Run profiling for one step
        start_time = time.time()
        prof = self.train_step_with_profiling(inp, label)
        elapsed_time = time.time() - start_time
        setup_time = start_setup_time - time.time()

        # Aggregate total FLOPs from the profiler
        cycle_flops = sum(
            e.flops
            for e in prof.events()
            if hasattr(e, "flops") and e.flops is not None
        )

        # Number of Training steps
        train_steps_per_ep = self.ld_train.batch_sampler.iters_per_ep
        max_train_steps = train_steps_per_ep * self.trainer_cfg.epochs

        # Compute stats
        batch_size = inp.shape[0]
        accumulation_steps = self.training_params.n_gradient_accumulation
        effective_batch_size = accumulation_steps * batch_size

        # Total FLOPs per sample
        flops_per_sample = cycle_flops / batch_size

        # PFLOPS calculations
        sample_pflops = flops_per_sample / (elapsed_time * 1e15)
        step_pflops = cycle_flops / (elapsed_time * 1e15)

        # Calculate full training cost
        total_training_flops = (
            cycle_flops * max_train_steps
        )  # Remove the accumulation_steps multiplication
        total_pflops = total_training_flops / 1e15

        # For reporting per-step metrics with accumulation
        effective_flops_per_step = (
            cycle_flops * accumulation_steps
        )  # Only for reporting what each logical step processes

        # Model Stats
        total_trainable_params = sum(
            p.numel() for p in self.var_wo_ddp.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.var_wo_ddp.parameters())
        param_efficiency = total_trainable_params / total_params * 100

        # Print results
        print(f"--- Profiling Results ---")
        print(f"Adaptation: {self.trainer_cfg.name}")
        print(f"Dataset: {self.dataset_cfg.name}")
        print(f"Model: {self.model_cfg.name}")
        print(f"Batch size (mini-batch): {batch_size}")
        print(f"Accumulation steps: {accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Total trainable parameters: {total_trainable_params}")
        print(f"Total parameters: {total_params}")
        print(f"Parameter efficiency: {param_efficiency:.2f}%")
        print(f"Mini-batch FLOPs (fwd+bwd): {cycle_flops:.2f}")
        print(f"Effective FLOPs per training step: {effective_flops_per_step:.2f}")
        print(f"Elapsed time for one mini-batch (s): {elapsed_time:.4f}")
        print(f"PFLOPs per sample: {sample_pflops:.4f}")
        print(f"PFLOPs per mini-batch: {step_pflops:.4f}")
        print(f"Total training steps: {max_train_steps}")
        print(f"Total training FLOPs: {total_training_flops:.2f}")
        print(f"Total training PFLOPs: {total_pflops:.4f}")
        print(f"--- End of Profiling Results ---")

        # Save results
        results = {
            "adaptation": self.trainer_cfg.name,
            "dataset": self.dataset_cfg.name,
            "model": self.model_cfg.name,
            "batch_size": int(batch_size),
            "accumulation_steps": int(accumulation_steps),
            "effective_batch_size": int(effective_batch_size),
            "total_trainable_params": int(total_trainable_params),
            "total_params": int(total_params),
            "parameter_efficiency": f"{param_efficiency:.2f}%",
            "mini_batch_flops": float(cycle_flops),
            "effective_flops_per_training_step": float(effective_flops_per_step),
            "elapsed_time_per_mini_batch_seconds": float(elapsed_time),
            "pflops_per_sample": float(sample_pflops),
            "pflops_per_mini_batch": float(step_pflops),
            "total_training_steps": int(max_train_steps),
            "total_training_flops": float(total_training_flops),
            "total_training_pflops": float(total_pflops),
        }

        output_dir = os.path.join(
            "out/profiler/",
            self.model_cfg.name,
            self.trainer_cfg.name,
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{self.dataset_cfg.name}_{self.finetuning_cfg.name}.json",
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Profiling results saved to {output_file}")

        print("\nProfiling completed.")


class FFTVARProfiler(ProfilingVARTrainer, FFTVARTrainer): ...


class LoraVARProfiler(ProfilingVARTrainer, LoraVARTrainer): ...


class LNTuningVARProfiler(ProfilingVARTrainer, LNTuningVARTrainer): ...
