# 遇到的问题和解答

## Environment
### Problem 1
#### Description
当直接在 envs 里修改 anaconda 的环境名时，终端输入 clear 会报错：

```bash
    terminals database is inaccessible
```
#### Solution
在终端输入：

```bash
    mv $CONDA_PREFIX/bin/clear $CONDA_PREFIX/bin/clear_old
```
即可。

### Problem 2
#### Description
连接学校校内访问 VPN 之后，由学校密码登录界面无需输入验证码确定是在校内，但仍然无法连接校内服务器。
#### Solution
CFW 打开 TUN Mode，介绍可以看这里：[TUN Mode](https://docs.gtk.pw/contents/tun.html)，其实就是对于不遵循系统代理的软件，TUN 模式可以接管其流量并交由 CFW 处理。
**TUN Mode需要先安装上方的Service Mode**，不然 TUN Mode 模式开了也没用。

