#!/usr/bin/env python3
"""
AI Training Simulator for Observability Demo
FOSDEM 2026 - HPC Devroom
Samuel Desseaux - Erythix

This simulator exposes metrics across all 3 observability layers:
- Layer 1: Infrastructure (GPU, memory, power)
- Layer 2: Workload Efficiency (throughput, I/O, communication)
- Layer 3: Model Health (loss, gradients, convergence)

Scenarios:
- normal: Everything works fine
- io_bottleneck: Data loading becomes slow (GPU high, throughput drops)
- gradient_explosion: Gradients explode during training
- loss_plateau: Model stops learning (loss stagnates)
"""

import os
import time
import random
import math
from prometheus_client import start_http_server, Gauge, Counter, Histogram
import threading

# ============================================
# METRICS DEFINITION
# ============================================

# --- Layer 1: Infrastructure Metrics ---
# Simulates DCGM-style GPU metrics
gpu_utilization = Gauge('dcgm_fi_dev_gpu_util', 
    'GPU utilization percentage', ['gpu', 'node'])
gpu_memory_used = Gauge('dcgm_fi_dev_fb_used', 
    'GPU memory used in MB', ['gpu', 'node'])
gpu_memory_total = Gauge('dcgm_fi_dev_fb_total', 
    'GPU memory total in MB', ['gpu', 'node'])
gpu_power_usage = Gauge('dcgm_fi_dev_power_usage', 
    'GPU power usage in watts', ['gpu', 'node'])
gpu_temperature = Gauge('dcgm_fi_dev_gpu_temp', 
    'GPU temperature in celsius', ['gpu', 'node'])
nvlink_bandwidth = Gauge('dcgm_fi_dev_nvlink_bandwidth_total', 
    'NVLink bandwidth in GB/s', ['gpu', 'node'])
# Advanced GPU metrics
gpu_sm_clock = Gauge('dcgm_fi_dev_sm_clock',
    'GPU SM clock frequency in MHz', ['gpu', 'node'])
gpu_memory_clock = Gauge('dcgm_fi_dev_mem_clock',
    'GPU memory clock frequency in MHz', ['gpu', 'node'])
gpu_pcie_tx = Gauge('dcgm_fi_dev_pcie_tx_throughput',
    'PCIe TX throughput in MB/s', ['gpu', 'node'])
gpu_pcie_rx = Gauge('dcgm_fi_dev_pcie_rx_throughput',
    'PCIe RX throughput in MB/s', ['gpu', 'node'])
gpu_tensor_active = Gauge('dcgm_fi_dev_tensor_active',
    'Tensor core utilization percentage', ['gpu', 'node'])
gpu_sm_active = Gauge('dcgm_fi_dev_sm_active',
    'SM activity percentage', ['gpu', 'node'])
gpu_dram_active = Gauge('dcgm_fi_dev_dram_active',
    'DRAM activity percentage', ['gpu', 'node'])
gpu_fp16_active = Gauge('dcgm_fi_dev_fp16_active',
    'FP16 engine utilization', ['gpu', 'node'])
gpu_ecc_errors = Counter('dcgm_fi_dev_ecc_dbe_total',
    'ECC double-bit errors', ['gpu', 'node'])

# --- Layer 2: Workload Efficiency Metrics ---
training_samples_per_second = Gauge('training_samples_per_second',
    'Training throughput in samples/second', ['job_id', 'model'])
training_tokens_per_second = Gauge('training_tokens_per_second',
    'Training throughput in tokens/second (for LLMs)', ['job_id', 'model'])
data_loading_time_ratio = Gauge('training_data_loading_ratio',
    'Ratio of time spent loading data vs training', ['job_id', 'model'])
all_reduce_time_seconds = Gauge('training_all_reduce_time_seconds',
    'Time spent in gradient synchronization', ['job_id', 'model'])
checkpoint_duration_seconds = Gauge('training_checkpoint_duration_seconds',
    'Time to save model checkpoint', ['job_id', 'model'])
batch_efficiency = Gauge('training_batch_efficiency',
    'Effective batch size vs configured (0-1)', ['job_id', 'model'])
gpu_memory_efficiency = Gauge('training_memory_efficiency',
    'Memory utilization efficiency', ['job_id', 'model'])
# Advanced workload metrics
training_step_duration = Gauge('training_step_duration_seconds',
    'Duration of each training step', ['job_id', 'model'])
training_forward_time = Gauge('training_forward_time_seconds',
    'Time spent in forward pass', ['job_id', 'model'])
training_backward_time = Gauge('training_backward_time_seconds',
    'Time spent in backward pass', ['job_id', 'model'])
training_optimizer_time = Gauge('training_optimizer_time_seconds',
    'Time spent in optimizer step', ['job_id', 'model'])
training_mfu = Gauge('training_model_flops_utilization',
    'Model FLOPs Utilization (MFU) percentage', ['job_id', 'model'])
training_hfu = Gauge('training_hardware_flops_utilization',
    'Hardware FLOPs Utilization (HFU) percentage', ['job_id', 'model'])
training_gpu_hours = Counter('training_gpu_hours_total',
    'Total GPU hours consumed', ['job_id', 'model'])
training_estimated_cost = Gauge('training_estimated_cost_usd',
    'Estimated training cost in USD', ['job_id', 'model'])
training_eta_seconds = Gauge('training_eta_seconds',
    'Estimated time to completion', ['job_id', 'model'])

# --- Layer 3: Model Health Metrics ---
training_loss = Gauge('training_loss',
    'Current training loss value', ['job_id', 'model', 'loss_type'])
validation_loss = Gauge('validation_loss',
    'Current validation loss value', ['job_id', 'model'])
gradient_norm = Gauge('training_gradient_norm',
    'L2 norm of gradients', ['job_id', 'model'])
learning_rate = Gauge('training_learning_rate',
    'Current learning rate', ['job_id', 'model'])
training_step = Counter('training_steps_total',
    'Total training steps completed', ['job_id', 'model'])
epoch_progress = Gauge('training_epoch_progress',
    'Current epoch progress (0-1)', ['job_id', 'model'])
convergence_velocity = Gauge('training_convergence_velocity',
    'Rate of loss decrease', ['job_id', 'model'])
# Advanced model health metrics
gradient_norm_per_layer = Gauge('training_gradient_norm_layer',
    'Gradient norm per layer', ['job_id', 'model', 'layer'])
weight_norm = Gauge('training_weight_norm',
    'L2 norm of model weights', ['job_id', 'model'])
activation_memory_mb = Gauge('training_activation_memory_mb',
    'Memory used by activations', ['job_id', 'model'])
parameter_count = Gauge('training_parameter_count',
    'Total trainable parameters', ['job_id', 'model'])
training_perplexity = Gauge('training_perplexity',
    'Training perplexity (for LLMs)', ['job_id', 'model'])
validation_accuracy = Gauge('validation_accuracy',
    'Validation accuracy', ['job_id', 'model'])
training_nan_count = Counter('training_nan_count_total',
    'Count of NaN values detected', ['job_id', 'model'])
training_overflow_count = Counter('training_overflow_count_total',
    'Count of overflow events', ['job_id', 'model'])
loss_spike_detected = Gauge('training_loss_spike_detected',
    'Binary flag for loss spike detection', ['job_id', 'model'])

# --- Meta Metrics ---
training_status = Gauge('training_status',
    'Training status (1=running, 0=stopped)', ['job_id', 'model', 'status'])

# ============================================
# SCENARIO CONFIGURATIONS
# ============================================

SCENARIOS = {
    'normal': {
        'description': 'Normal training - all metrics healthy',
        'gpu_util_range': (85, 95),
        'throughput_base': 1000,
        'throughput_variance': 50,
        'data_loading_ratio': 0.05,
        'gradient_norm_range': (0.5, 2.0),
        'loss_decay_rate': 0.995,
        'initial_loss': 2.5,
    },
    'io_bottleneck': {
        'description': 'I/O bottleneck - GPU high but throughput drops',
        'gpu_util_range': (88, 94),  # GPU looks fine!
        'throughput_base': 600,  # But throughput is low
        'throughput_variance': 100,
        'data_loading_ratio': 0.45,  # Data loading is the problem
        'gradient_norm_range': (0.5, 2.0),
        'loss_decay_rate': 0.997,  # Learning is slower
        'initial_loss': 2.5,
    },
    'gradient_explosion': {
        'description': 'Gradient explosion - training becomes unstable',
        'gpu_util_range': (90, 98),
        'throughput_base': 950,
        'throughput_variance': 80,
        'data_loading_ratio': 0.08,
        'gradient_norm_range': (50, 500),  # Gradients explode!
        'loss_decay_rate': 1.02,  # Loss actually increases
        'initial_loss': 2.5,
    },
    'loss_plateau': {
        'description': 'Loss plateau - model stops improving',
        'gpu_util_range': (85, 92),
        'throughput_base': 980,
        'throughput_variance': 40,
        'data_loading_ratio': 0.06,
        'gradient_norm_range': (0.01, 0.05),  # Vanishing gradients
        'loss_decay_rate': 0.9999,  # Almost no improvement
        'initial_loss': 1.2,
    },
}

# ============================================
# SIMULATOR CLASS
# ============================================

class TrainingSimulator:
    def __init__(self, scenario='normal', num_gpus=4, job_id='job_001', model='resnet50'):
        self.scenario = SCENARIOS.get(scenario, SCENARIOS['normal'])
        self.scenario_name = scenario
        self.num_gpus = num_gpus
        self.job_id = job_id
        self.model = model
        self.node = 'hpc-node-01'
        
        self.step = 0
        self.current_loss = self.scenario['initial_loss']
        self.epoch = 0
        self.steps_per_epoch = 1000
        self.running = True
        
        # For I/O bottleneck scenario - simulate gradual degradation
        self.io_degradation_factor = 1.0
        
        print(f"Starting simulator with scenario: {scenario}")
        print(f"Description: {self.scenario['description']}")
        
    def add_noise(self, value, variance_pct=0.05):
        """Add realistic noise to a metric."""
        return value * (1 + random.uniform(-variance_pct, variance_pct))
    
    def update_infrastructure_metrics(self):
        """Update Layer 1: Infrastructure metrics."""
        for gpu_id in range(self.num_gpus):
            gpu_label = f'gpu{gpu_id}'
            
            # GPU Utilization
            base_util = random.uniform(*self.scenario['gpu_util_range'])
            gpu_utilization.labels(gpu=gpu_label, node=self.node).set(base_util)
            
            # GPU Memory (24GB GPUs simulated)
            total_mem = 24576  # 24GB in MB
            used_mem = total_mem * random.uniform(0.75, 0.92)
            gpu_memory_used.labels(gpu=gpu_label, node=self.node).set(used_mem)
            gpu_memory_total.labels(gpu=gpu_label, node=self.node).set(total_mem)
            
            # Power usage (300-400W typical for training)
            power = self.add_noise(350, 0.1)
            gpu_power_usage.labels(gpu=gpu_label, node=self.node).set(power)
            
            # Temperature (65-80°C normal)
            temp = self.add_noise(72, 0.08)
            gpu_temperature.labels(gpu=gpu_label, node=self.node).set(temp)
            
            # NVLink bandwidth
            nvlink = self.add_noise(40, 0.15)  # ~40 GB/s
            nvlink_bandwidth.labels(gpu=gpu_label, node=self.node).set(nvlink)
            
            # Advanced GPU metrics
            sm_clock = self.add_noise(1410, 0.05)  # MHz
            gpu_sm_clock.labels(gpu=gpu_label, node=self.node).set(sm_clock)
            
            mem_clock = self.add_noise(1215, 0.03)  # MHz
            gpu_memory_clock.labels(gpu=gpu_label, node=self.node).set(mem_clock)
            
            # PCIe throughput
            pcie_tx = self.add_noise(12000, 0.2)  # MB/s
            pcie_rx = self.add_noise(11500, 0.2)
            gpu_pcie_tx.labels(gpu=gpu_label, node=self.node).set(pcie_tx)
            gpu_pcie_rx.labels(gpu=gpu_label, node=self.node).set(pcie_rx)
            
            # Tensor core utilization
            tensor_util = base_util * random.uniform(0.7, 0.9)  # Usually lower than GPU util
            gpu_tensor_active.labels(gpu=gpu_label, node=self.node).set(tensor_util)
            
            # SM activity
            sm_active = base_util * random.uniform(0.85, 0.98)
            gpu_sm_active.labels(gpu=gpu_label, node=self.node).set(sm_active)
            
            # DRAM activity - higher when memory bound
            dram_active = self.add_noise(65, 0.15)
            if self.scenario_name == 'io_bottleneck':
                dram_active = self.add_noise(45, 0.2)  # Lower when waiting for data
            gpu_dram_active.labels(gpu=gpu_label, node=self.node).set(dram_active)
            
            # FP16 utilization
            fp16_active = tensor_util * random.uniform(0.8, 1.0)
            gpu_fp16_active.labels(gpu=gpu_label, node=self.node).set(fp16_active)
    
    def update_workload_metrics(self):
        """Update Layer 2: Workload Efficiency metrics."""
        labels = {'job_id': self.job_id, 'model': self.model}
        
        # Throughput with scenario-specific behavior
        if self.scenario_name == 'io_bottleneck' and self.step > 50:
            # Gradual I/O degradation
            self.io_degradation_factor = max(0.4, self.io_degradation_factor - 0.005)
            throughput = self.scenario['throughput_base'] * self.io_degradation_factor
        else:
            throughput = self.scenario['throughput_base']
        
        throughput = self.add_noise(throughput, 
            self.scenario['throughput_variance'] / self.scenario['throughput_base'])
        
        training_samples_per_second.labels(**labels).set(throughput)
        training_tokens_per_second.labels(**labels).set(throughput * 512)  # Assume 512 tokens/sample
        
        # Data loading ratio
        data_ratio = self.scenario['data_loading_ratio']
        if self.scenario_name == 'io_bottleneck' and self.step > 50:
            data_ratio = min(0.6, data_ratio + (self.step - 50) * 0.005)
        data_loading_time_ratio.labels(**labels).set(self.add_noise(data_ratio, 0.1))
        
        # All-reduce time (gradient sync)
        all_reduce = self.add_noise(0.05, 0.2)  # 50ms typical
        all_reduce_time_seconds.labels(**labels).set(all_reduce)
        
        # Checkpoint duration (every 100 steps)
        if self.step % 100 < 5:
            checkpoint_duration_seconds.labels(**labels).set(self.add_noise(15, 0.3))
        else:
            checkpoint_duration_seconds.labels(**labels).set(0)
        
        # Batch efficiency
        batch_eff = 0.95 if self.scenario_name != 'io_bottleneck' else 0.7
        batch_efficiency.labels(**labels).set(self.add_noise(batch_eff, 0.05))
        
        # Memory efficiency
        mem_eff = self.add_noise(0.88, 0.05)
        gpu_memory_efficiency.labels(**labels).set(mem_eff)
        
        # Advanced workload metrics
        step_duration = 1.0 / throughput * 32  # Assuming batch size 32
        training_step_duration.labels(**labels).set(self.add_noise(step_duration, 0.1))
        
        # Time breakdown
        forward_time = step_duration * 0.35
        backward_time = step_duration * 0.45
        optimizer_time = step_duration * 0.10
        training_forward_time.labels(**labels).set(self.add_noise(forward_time, 0.1))
        training_backward_time.labels(**labels).set(self.add_noise(backward_time, 0.1))
        training_optimizer_time.labels(**labels).set(self.add_noise(optimizer_time, 0.15))
        
        # MFU/HFU - Model and Hardware FLOPs Utilization
        base_mfu = 45 if self.scenario_name == 'normal' else 30
        if self.scenario_name == 'io_bottleneck':
            base_mfu = max(20, base_mfu * self.io_degradation_factor)
        training_mfu.labels(**labels).set(self.add_noise(base_mfu, 0.1))
        training_hfu.labels(**labels).set(self.add_noise(base_mfu * 1.15, 0.1))
        
        # GPU hours consumed (4 GPUs)
        training_gpu_hours.labels(**labels).inc(4 / 3600)  # 4 GPUs, 1 second
        
        # Estimated cost ($2/GPU-hour for A100)
        gpu_hours_total = self.step * 4 / 3600
        training_estimated_cost.labels(**labels).set(gpu_hours_total * 2.0)
        
        # ETA calculation
        total_steps = 10000
        remaining_steps = max(0, total_steps - self.step)
        eta_seconds = remaining_steps * step_duration
        training_eta_seconds.labels(**labels).set(eta_seconds)
    
    def update_model_health_metrics(self):
        """Update Layer 3: Model Health metrics."""
        labels = {'job_id': self.job_id, 'model': self.model}
        
        # Loss calculation
        if self.scenario_name == 'gradient_explosion' and self.step > 30:
            # Loss explodes
            self.current_loss *= 1.05
            self.current_loss = min(self.current_loss, 100)  # Cap it
        elif self.scenario_name == 'loss_plateau':
            # Very slow decay
            self.current_loss *= self.scenario['loss_decay_rate']
            self.current_loss = max(self.current_loss, 1.15)  # Plateau
        else:
            # Normal decay
            self.current_loss *= self.scenario['loss_decay_rate']
            self.current_loss = max(self.current_loss, 0.1)
        
        loss_value = self.add_noise(self.current_loss, 0.02)
        training_loss.labels(loss_type='cross_entropy', **labels).set(loss_value)
        
        # Validation loss (slightly higher than training)
        val_loss = loss_value * self.add_noise(1.1, 0.05)
        validation_loss.labels(**labels).set(val_loss)
        
        # Gradient norm
        grad_min, grad_max = self.scenario['gradient_norm_range']
        if self.scenario_name == 'gradient_explosion' and self.step > 30:
            # Exponential growth
            grad_norm = min(1000, grad_min * (1.1 ** (self.step - 30)))
        else:
            grad_norm = random.uniform(grad_min, grad_max)
        gradient_norm.labels(**labels).set(grad_norm)
        
        # Learning rate (cosine annealing simulation)
        base_lr = 0.001
        lr = base_lr * (1 + math.cos(math.pi * self.step / 1000)) / 2
        learning_rate.labels(**labels).set(lr)
        
        # Training progress
        training_step.labels(**labels).inc()
        epoch_progress.labels(**labels).set((self.step % self.steps_per_epoch) / self.steps_per_epoch)
        
        # Convergence velocity (loss change rate)
        velocity = -math.log(self.scenario['loss_decay_rate']) * 100  # Positive = improving
        if self.scenario_name == 'gradient_explosion':
            velocity = -5  # Negative = getting worse
        convergence_velocity.labels(**labels).set(velocity)
        
        # Training status
        training_status.labels(status='running', **labels).set(1)
        training_status.labels(status='error', **labels).set(
            1 if (self.scenario_name == 'gradient_explosion' and self.step > 100) else 0
        )
        
        # Advanced model health metrics
        
        # Per-layer gradient norms (simulate 12 layers)
        for layer_idx in range(12):
            layer_name = f'layer_{layer_idx}'
            layer_grad = grad_norm * random.uniform(0.5, 1.5)
            gradient_norm_per_layer.labels(layer=layer_name, **labels).set(layer_grad)
        
        # Weight norm
        weight_norm.labels(**labels).set(self.add_noise(150, 0.05))
        
        # Activation memory
        act_mem = self.add_noise(8500, 0.1)  # MB
        activation_memory_mb.labels(**labels).set(act_mem)
        
        # Parameter count (7B params for llama2_7b)
        parameter_count.labels(**labels).set(7_000_000_000)
        
        # Perplexity (exp of loss for LLMs)
        perplexity = math.exp(min(loss_value, 10))  # Cap to avoid overflow
        training_perplexity.labels(**labels).set(perplexity)
        
        # Validation accuracy
        base_accuracy = 0.85 - (loss_value * 0.1)
        validation_accuracy.labels(**labels).set(max(0, min(1, self.add_noise(base_accuracy, 0.02))))
        
        # Loss spike detection
        spike_detected = 0
        if self.scenario_name == 'gradient_explosion' and self.step > 50:
            spike_detected = 1
        loss_spike_detected.labels(**labels).set(spike_detected)
    
    def run_step(self):
        """Execute one training step simulation."""
        self.update_infrastructure_metrics()
        self.update_workload_metrics()
        self.update_model_health_metrics()
        
        self.step += 1
        if self.step % self.steps_per_epoch == 0:
            self.epoch += 1
            print(f"Epoch {self.epoch} completed (scenario: {self.scenario_name})")
    
    def run(self, interval=1.0):
        """Run the simulator continuously."""
        while self.running:
            self.run_step()
            time.sleep(interval)


def main():
    # Get scenario from environment
    scenario = os.environ.get('SCENARIO', 'normal')
    port = int(os.environ.get('METRICS_PORT', '9100'))
    
    # Start Prometheus metrics server
    print(f"Starting metrics server on port {port}...")
    start_http_server(port)
    
    # Create and run simulator
    simulator = TrainingSimulator(
        scenario=scenario,
        num_gpus=4,
        job_id='fosdem_demo_001',
        model='llama2_7b'
    )
    
    print(f"\n{'='*50}")
    print(f"AI Training Simulator - FOSDEM 2026 Demo")
    print(f"Scenario: {scenario}")
    print(f"Metrics available at: http://localhost:{port}/metrics")
    print(f"{'='*50}\n")
    
    try:
        simulator.run(interval=1.0)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == '__main__':
    main()
