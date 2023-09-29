import torch
from typing import List


def main(M_number, C_number, boat_capacity):
    """
    构建符合条件的有向有环图,使用邻接矩阵来存储
    然后使用DFS来搜索解空间
    Args:
        M_number (int): 传教士的数量
        C_number (int): 野人的数量
        boat_capacity (int): 船能容纳的人数
    """
    if M_number <= 0 or C_number <= 0 or boat_capacity <= 0:
        raise ValueError

    init_states = torch.tensor(
        [[3, 3, 0]], dtype=torch.int
    )
    goal_states = torch.tensor(
        [[0, 0, 1]], dtype=torch.int
    )
    operators = generate_operators(M_number, C_number, boat_capacity)

    # 图初始化
    possible_state_number = (M_number+1)*(C_number+1)*2
    graph = torch.zeros((possible_state_number, possible_state_number), dtype=torch.int)
    graph_states = []
    for i in range(init_states.size(0)):
        graph_states.append(init_states[i].tolist())
    print(graph_states)

    # 寻找符合条件的初始状态与构造邻接矩阵
    index = 0
    total_states = init_states.size(0)
    while index < total_states:
        # 检查operators能否作用于index状态
        for operator in operators:
            start_state = torch.tensor(graph_states[index], dtype=torch.int)
            end_state = state_move(start_state, operator)
            if end_state is not None:
                if state_check(end_state, M_number, C_number):
                    print(f"{start_state.tolist()}-{operator.tolist()}->{end_state.tolist()} success")
                    # 还需要去重
                    if end_state.tolist() not in graph_states:
                        graph_states.append(end_state.tolist())
                        graph[index][total_states] = 1
                        total_states += 1
                    else:
                        # 虽然end_state重复了，不参与新状态的增加，但对邻接矩阵有影响
                        graph[index][graph_states.index(end_state.tolist())] = 1
                else:
                    print(f"{start_state.tolist()}-{operator.tolist()}->{end_state.tolist()} fail")

        index += 1
        print()

    # 打印这个图
    for i in range(graph.size(0)):
        for j in range(graph.size(1)):
            if graph[i][j]:
                print(f"{graph_states[i]}=>{graph_states[j]}")

    # 打印节点信息
    for i, graph_state in enumerate(graph_states):
        print(i, graph_state)

    # 遍历有向有环图
    paths = []
    for init_state in init_states:
        start_index = graph_states.index(init_state.tolist())
        for goal_state in goal_states:
            end_index = graph_states.index(goal_state.tolist())
            DFS(start_index, end_index, graph, [start_index], paths)
    print(paths)

    for path in paths:
        for node in path:
            print(f"{graph_states[node]}", end="=>")
        print()


def DFS(pwd_index, end_index, graph: torch.tensor, path: list, paths: List[list]):
    if pwd_index == end_index:
        paths.append(path.copy())
        return

    for i in range(graph[pwd_index].size(0)):
        if graph[pwd_index][i] == 1:
            if i in path:
                continue
            path.append(i)
            DFS(i, end_index, graph, path, paths)
            path.pop()


def state_check(state: torch.Tensor, M_number: int, C_number: int):
    # 不能有负数，当左岸传教士不为0时，左岸传教士的数量要大于等于左岸野人数量；当右岸传教士不为0时，右岸传教士的数量要大于等于右岸野人数量。
    # lm>=lc,rm>=rc, lm+rm=m, lc+rc=c
    # m-lm>=c-lc
    if (state < 0).any():
        return False
    if M_number-state[0] < 0 or C_number-state[1] < 0:
        return False
    if state[0] != 0 and state[0] < state[1]:
        return False
    if M_number - state[0] != 0 and ((M_number-state[0]) < (C_number-state[1])):
        return False

    return True


def state_move(start_state: torch.Tensor, operator: torch.Tensor):
    """
    状态移动
    Args:
        start_state (torch.Tensor): 起始状态
        operator (torch.Tensor): 操作符

    Returns:
        end_state (torch.Tensor): 终止状态
    """
    # 左=》右
    if start_state[2] == 0 and operator[2] == 1:
        end_state = start_state - operator
        end_state[2] = 1
        end_state = end_state.to(torch.int)
    # 右=》左
    elif start_state[2] == 1 and operator[2] == 0:
        end_state = start_state+operator
        end_state[2] = 0
        end_state = end_state.to(torch.int)
    else:
        end_state = None
    return end_state


def generate_operators(M_number, C_number, boat_capacity):
    """
    构建符合条件的操作符
    约束条件: 船上传教士不为0时, 传教士>=野人>=0, 且0<传教士+野人<=boat_capacity
    传教士为0时, boat_capacity>=野人>0
    Args:
        M_number (int): 传教士的数量
        C_number (int): 野人的数量
        boat_capacity (int): 船能容纳的人数
    """
    operators = []
    for d in range(2):
        for m in range(M_number+1):
            for c in range(C_number+1):
                if m != 0 and m >= c and m+c > 0 and m+c <= boat_capacity:
                    operators.append([m, c, d])
                if m == 0 and c > 0 and c <= boat_capacity:
                    operators.append([m, c, d])

    print("operators", operators)
    return torch.tensor(operators, dtype=torch.int)


if __name__ == "__main__":
    main(3, 3, 2)
