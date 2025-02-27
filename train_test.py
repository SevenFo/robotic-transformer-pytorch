import torch
import unittest
from train import LitRT1


class TestNormalizeAction(unittest.TestCase):
    def setUp(self):
        # 设置测试环境
        self.num_bins = 100
        # 创建一个虚拟的动作数据，形状为[1, 13]
        # 假设前6个为关节角度，第7个为末端执行器关节，8-10为末端执行器xyz位置，11-13为末端执行器rpy旋转
        self.action = torch.tensor(
            [
                # 关节角度（6个）
                [
                    -torch.pi / 2,
                    0,
                    torch.pi / 2,
                    -torch.pi,
                    torch.pi,
                    0,
                    # ee_joint (第7个元素，索引为7)
                    0.7,
                    # ee_xyz (第8-10个元素，索引为8-10)
                    -0.4,
                    0.4,
                    0,
                    # ee_rpy (第11-13个元素，索引为11-13)
                    torch.pi / 4,
                    -torch.pi / 4,
                    0,
                ]
            ]
        )

    def test_normalize_action(self):
        """测试动作归一化函数"""
        normalized = LitRT1._normalize_action(self.action, self.num_bins)

        # 检查输出形状
        self.assertEqual(normalized.shape, torch.Size([1, 13]))

        # 检查输出值是否在[0, num_bins-1]范围内
        self.assertTrue((normalized >= 0).all().item())
        self.assertTrue((normalized < self.num_bins).all().item())

        # 检查关节角度归一化 (应该是[-π, π]范围归一化到[0, num_bins-1])
        # -π/2对应约为25% -> ~25
        self.assertAlmostEqual(normalized[0, 0].item(), 25, delta=1)
        # 0对应约为50% -> ~50
        self.assertAlmostEqual(normalized[0, 1].item(), 50, delta=1)
        # π/2对应约为75% -> ~75
        self.assertAlmostEqual(normalized[0, 2].item(), 75, delta=1)

        # 检查ee_joint归一化 (应该是[0, 1.0]范围归一化到[0, num_bins-1])
        # 0.7对应约70% -> ~70
        self.assertAlmostEqual(normalized[0, 6].item(), 70, delta=1)

        # 检查ee_xyz归一化 (应该是[-0.8, 0.8]范围归一化到[0, num_bins-1])
        # -0.4对应约为25% -> ~25
        self.assertAlmostEqual(normalized[0, 7].item(), 25, delta=1)
        # 0.4对应约为75% -> ~75
        self.assertAlmostEqual(normalized[0, 8].item(), 75, delta=1)

    # def test_edge_cases(self):
    #     """测试边缘情况"""
    #     # 创建一个超出范围的动作
    #     out_of_range = torch.tensor(
    #         [[2 * torch.pi, -2 * torch.pi, 0, 0, 0, 0, 0, 2.0, 1.0, 1.0, 0, 0, 0, 0]]
    #     )
    #     normalized = LitRT1._normalize_action(out_of_range, self.num_bins)

    #     # 检查是否被正确裁剪到范围内
    #     self.assertTrue((normalized >= 0).all().item())
    #     self.assertTrue((normalized < self.num_bins).all().item())

    def test_batch_normalization(self):
        """测试批量归一化"""
        batch_size = 3
        batch_action = self.action.repeat(batch_size, 1)
        normalized = LitRT1._normalize_action(batch_action, self.num_bins)

        # 检查输出形状
        self.assertEqual(normalized.shape, torch.Size([batch_size, 13]))

        # 检查所有批次的结果是否相同
        for i in range(1, batch_size):
            self.assertTrue(torch.all(normalized[0] == normalized[i]).item())


if __name__ == "__main__":
    unittest.main()
