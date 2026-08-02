# 使用 Isaac Sim 将 URDF 转成 USD

本目录中的转换器调用 Isaac Sim URDF Importer 的
`URDFCreateImportConfig` 和 `URDFParseAndImportFile` 接口，并针对 Lula Robot
Description Editor 做了两项处理：

1. 在临时 URDF 中把 `package://HDJQR-0102-0055.SLDASM/meshes/...` 解析到本目录的
   `meshes/`；原始 URDF 不会被修改。
2. 导入后取消网格引用的 instanceable 标记。Lula 编辑器不能使用 instance proxy
   自动生成碰撞球，因此这一步不能省略。

## 转换

本机默认使用 `/home/z017/isaacsim`：

```bash
./convert_urdf_to_usd.sh
```

该机器人是无人机，因此转换器默认生成 **floating base**，不会创建世界坐标系到
`base_link` 的固定关节。如果确实需要固定基座，可显式添加 `--fixed-base`。

默认主文件输出到：

```text
usd/HDJQR-0102-0055.SLDASM/HDJQR-0102-0055.SLDASM.usd
```

同时会生成 `configuration/`（部分 URDF 还会生成 `meshes/`）等分层资产；使用或
移动资产时要保留整个 `usd/HDJQR-0102-0055.SLDASM/` 目录。再次转换需加
`--overwrite`：

```bash
./convert_urdf_to_usd.sh --overwrite
```

其他 Isaac Sim 安装位置：

```bash
ISAAC_SIM_PATH=/path/to/isaac-sim ./convert_urdf_to_usd.sh
```

## 在 Lula Robot Description Editor 中使用

1. 在 Isaac Sim 中打开或引用上面的主 USD 文件。
2. 点击 **Play**。
3. 打开 **Lula Robot Description Editor**。
4. 在 **Selection Panel** 中选择该 Articulation，再逐个选择 Link。
5. 在 **Link Sphere Editor / Editor Tools** 中自动生成或手动调整碰撞球。
6. 设置 Active Joints、默认关节位置以及加速度/jerk 限制，然后在
   **Export to Lula Robot Description File** 中保存 YAML。

原先位于 `urdf/HDJQR-0102-0055.SLDASM/` 的同名 USD 没有成功导入 STL 网格，
不要将它用于自动生成碰撞球。
