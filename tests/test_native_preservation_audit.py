import torch
import pytest
from slotspe_proof.native_preservation_audit import preservation_case, summary_dependence_counterexample, actual_hook_collision, cancellation_case


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]): yield
    torch.set_num_threads(threads)


def test_native_size_source_adapter_has_lost_contrast_collision():
    row = preservation_case()
    assert row['encoder_ambient_dimension'] == 2048
    assert row['encoder_structural_rank_upper_bound'] == 1600
    assert row['encoder_nullity_numeric'] >= 448
    assert row['replacement_max_difference'] < 1e-12
    assert row['permutation_invariant_native_moment_gap_mean'] == pytest.approx(.25)
    assert abs(row['content_loss_difference']) < 1e-12
    assert abs(row['hsic_difference']) < 1e-12


def test_bounded_residual_and_identical_slot_manifold_are_recoverable():
    row = preservation_case(8, 2, 6, 1, 4)
    assert row['bounded_residual_contraction_factor'] <= .5
    assert row['bounded_residual_inverse_max_error'] < 1e-10
    assert row['identical_slot_manifold_inverse_max_error'] < 1e-9
    assert row['bounded_residual_lower_distance_ratio'] == pytest.approx(1.)


def test_source_summary_hsic_can_miss_a_whole_shared_bit():
    row = summary_dependence_counterexample()
    assert abs(row['source_summary_hsic']) < 1e-12
    assert row['known_mi_shared_full_private_nats'] == pytest.approx(float(torch.log(torch.tensor(2.))))


def test_actual_replacement_hook_erases_generated_invariant_predictor():
    row = actual_hook_collision()
    assert row['hook_records_present']
    assert row['generated_native_predictor_gap'] == pytest.approx(.25)
    assert row['generated_replaced_predictor_gap'] < 1e-12


def test_unbounded_native_residual_can_cancel_input():
    row = cancellation_case()
    assert row['ordinary_residual_gap'] == 0
    assert row['half_gain_residual_gap'] == pytest.approx(row['input_gap']/2)


def test_zero_gain_and_zero_branch_have_a_dead_prediction_gradient():
    gain = torch.tensor(0., requires_grad=True); branch = torch.tensor(0., requires_grad=True)
    ((2+gain*branch)-1).square().backward()
    assert gain.grad == 0 and branch.grad == 0
    gain = torch.tensor(0., requires_grad=True); branch = torch.tensor(2., requires_grad=True)
    ((2+gain*branch)-1).square().backward()
    assert gain.grad != 0 and branch.grad == 0


def test_inverse_reconstruction_can_remove_task_gradient_to_partition():
    angle = torch.tensor(.7, dtype=torch.double, requires_grad=True)
    rotation = torch.stack([torch.stack([angle.cos(), -angle.sin()]), torch.stack([angle.sin(), angle.cos()])])
    source = torch.tensor([1., 2.], dtype=torch.double)
    code = rotation@source
    reconstructed = rotation.T@code
    reconstructed.square().sum().backward()
    assert torch.allclose(reconstructed, source, atol=1e-12)
    assert abs(float(angle.grad)) < 1e-12
