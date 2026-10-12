# CasADi 在数值分析组件之前初始化，求解过程使用独立进程。
import casadi

from revised_route_comparison import OUTPUT, solve


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    solve()
