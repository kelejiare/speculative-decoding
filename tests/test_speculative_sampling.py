import unittest

import torch

from minicpmo_dflash.speculative import (
    DFlashSamplingConfig,
    speculative_accept_sample,
)


class SpeculativeSamplingTest(unittest.TestCase):
    def test_sampled_q_preserves_target_distribution(self) -> None:
        p = torch.tensor([0.10, 0.30, 0.60])
        q = torch.tensor([0.50, 0.25, 0.25])
        proposal_rng = torch.Generator().manual_seed(7)
        verifier_rng = torch.Generator().manual_seed(11)
        counts = torch.zeros(3, dtype=torch.long)
        accepted = 0
        draws = 20_000

        for _ in range(draws):
            candidate = int(torch.multinomial(q, 1, generator=proposal_rng).item())
            emitted, was_accepted = speculative_accept_sample(
                p, candidate, verifier_rng, q
            )
            counts[emitted] += 1
            accepted += int(was_accepted)

        empirical = counts.float() / draws
        expected_acceptance = torch.minimum(p, q).sum().item()
        self.assertTrue(torch.allclose(empirical, p, atol=0.015), empirical)
        self.assertAlmostEqual(accepted / draws, expected_acceptance, delta=0.015)

    def test_argmax_proposal_is_sample_and_compare(self) -> None:
        p = torch.tensor([0.15, 0.25, 0.60])
        rng = torch.Generator().manual_seed(23)
        counts = torch.zeros(3, dtype=torch.long)
        accepted = 0
        draws = 12_000
        for _ in range(draws):
            emitted, was_accepted = speculative_accept_sample(p, 2, rng)
            counts[emitted] += 1
            accepted += int(was_accepted)

        empirical = counts.float() / draws
        self.assertTrue(torch.allclose(empirical, p, atol=0.02), empirical)
        self.assertAlmostEqual(accepted / draws, p[2].item(), delta=0.02)

    def test_configuration_bounds(self) -> None:
        DFlashSamplingConfig(draft_temperature=-1, accept_top_k=64, accept_min_prob=1)
        with self.assertRaises(ValueError):
            DFlashSamplingConfig(draft_temperature=-1.1)
        with self.assertRaises(ValueError):
            DFlashSamplingConfig(draft_temperature=-0.5)
        with self.assertRaises(ValueError):
            DFlashSamplingConfig(accept_top_k=65)
        with self.assertRaises(ValueError):
            DFlashSamplingConfig(accept_min_prob=1.01)
        with self.assertRaises(ValueError):
            DFlashSamplingConfig(draft_temperature=float("nan"))


if __name__ == "__main__":
    unittest.main()
