# Pacman-Capture-the-Flag

本项目是基于 UC Berkeley CS188 课程的多人对战版吃豆人（Pacman）游戏，采用“夺旗”模式，旨在开发智能代理（AI agents）以实现协同作战。

## 🎯 项目简介

在本游戏中，两个团队各由两名智能代理组成，分别担任进攻和防守角色。进攻代理负责穿越地图，收集对方阵地的食物；防守代理则保护己方阵地，阻止对方代理的入侵。项目的目标是设计和实现高效的 AI 策略，使团队在比赛中取得优势。

## 🧠 技术实现

本项目主要使用以下人工智能技术：

- **启发式搜索（Heuristic Search）**：评估当前状态下的可能动作，引导代理决策。
- **蒙特卡罗树搜索（Monte Carlo Tree Search, MCTS）**：通过模拟多种游戏过程评估动作潜在收益。
- **粒子滤波（Particle Filtering）**：估计敌方代理位置，提高感知能力。
- **强化学习（Reinforcement Learning）**：尝试学习优化代理策略。

## 🧩 项目结构

```text
Pacman-Capture-the-Flag/
├── .idea/                 # 项目配置文件
├── img/                   # 图片资源
├── src/                   # 源代码目录
│   ├── capture.py         # 游戏主程序
│   ├── captureAgents.py   # 代理基类定义
│   ├── myTeam.py          # 自定义团队代理实现
│   └── ...                # 其他辅助模块
├── requirements.txt       # 项目依赖列表
├── runAll.sh              # 批量运行脚本
└── README.md              # 项目说明文件
```

## 🚀 使用说明
环境配置
操作系统：建议使用 Linux 或 macOS

Python 版本：3.6 或以上



## 📸 项目演示

## 📚 参考资料
UC Berkeley CS188 Pacman 项目

Monte Carlo Tree Search

Heuristic Search

## ❤️ 致谢
感谢 UC Berkeley 提供的 Pacman 项目框架，以及所有为该项目做出贡献的开发者和研究者。

如果您有任何问题或建议，欢迎提交 Issue 或 Pull Request。
