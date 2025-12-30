import numpy as np
from helpers import Environment, ActionMode
import pickle

def compare_policies(student_answer, solution_possibilities):
    return np.all(np.any(student_answer[:, :, None] == solution_possibilities, axis=2))


def get_test_1_environments():
    global ONE_BY_TWO_DYNAMICS, TWO_BY_TWO_DYNAMICS, FIVE_BY_FIVE_DYNAMICS

    with open("dynamics.pkl", "rb") as f:
        data = pickle.load(f)
        ONE_BY_TWO_DYNAMICS = data["ONE_BY_TWO_DYNAMICS"]
        TWO_BY_TWO_DYNAMICS = data["TWO_BY_TWO_DYNAMICS"]

        FIVE_BY_FIVE_DYNAMICS = data["FIVE_BY_FIVE_DYNAMICS"]

    ########## 1x2 board ###############
    one_by_two_board = np.array([[ 10, -10]])

    one_by_two_env = Environment(1, 2, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-10, goal_value=10)
    one_by_two_env.board = one_by_two_board # overwrite board
    one_by_two_env.dynamics = ONE_BY_TWO_DYNAMICS.copy()

    ########## 2x2 board ###############
    two_by_two_board = np.array([[-5, 50], [-1, -1]])

    two_by_two_env = Environment(2, 2, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=50)
    two_by_two_env.board = two_by_two_board  # overwrite board
    two_by_two_env.dynamics = TWO_BY_TWO_DYNAMICS.copy()

    ########## 5x5 board ###############
    five_by_five_board = np.array([
        [-1., -1., -1., -1., -1.],
        [-1., -5., -1., -1., -1.],
        [5., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.]
        ])
    
    five_by_five_env =  Environment(5, 5, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    five_by_five_env.board =five_by_five_board  # overwrite board
    five_by_five_env.dynamics = FIVE_BY_FIVE_DYNAMICS.copy()

    return one_by_two_env, two_by_two_env, five_by_five_env

def get_test_2_environments():
    global CARDINAL_THREE_BY_THREE_DYNAMICS, CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS

    with open("dynamics.pkl", "rb") as f:
        data = pickle.load(f)
        CARDINAL_THREE_BY_THREE_DYNAMICS = data["CARDINAL_THREE_BY_THREE_DYNAMICS"]
        CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS = data["CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS"]
        
    ########## Cardinal 3x3 Board ###############    
    three_by_three_board = np.array([[-1, 5, -1.],
                                    [-1, -5, -1.],
                                    [-1, -1, -1.]])

    cardinal_action_mode_env = Environment(3, 3, ActionMode.CARDINAL, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    cardinal_action_mode_env.board = three_by_three_board
    cardinal_action_mode_env.dynamics = CARDINAL_THREE_BY_THREE_DYNAMICS.copy()

    ########## 5x5 Continuous Board ###############
    five_by_five_board = np.array([[-0.8253895, -0.69805011, 0.25353471, 0.87788343, 0.53303922],
                                [ 0.04398602,  0.53420759, -0.50640458, -0.26112835,  0.03275385],
                                [-0.94955295,  0.98347051, -0.19441639,  0.15650246,  0.57653413],
                                [-0.38340712,  0.47741138, -0.36854612,  0.95404696, -0.76899452],
                                [-0.91146198, -0.56136624,  0.56397481, -0.94386589,  0.35540827]])

    continuous_rewards_env = Environment(5, 5, ActionMode.SIMPLE, discrete_rewards=False, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    continuous_rewards_env.board = five_by_five_board
    continuous_rewards_env.dynamics = CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS.copy()

    return cardinal_action_mode_env, continuous_rewards_env

def test_value_iteration_1(value_iteration_func):
    with open("dynamics.pkl", "rb") as f:
        data = pickle.load(f)
        ONE_BY_TWO_DYNAMICS = data["ONE_BY_TWO_DYNAMICS"]
        TWO_BY_TWO_DYNAMICS = data["TWO_BY_TWO_DYNAMICS"]

        FIVE_BY_FIVE_DYNAMICS = data["FIVE_BY_FIVE_DYNAMICS"]
    ########## Test 1x2 board ###############
    one_by_two_board = np.array([[ 10, -10]])
    one_by_two_board_value_iteration_solution = np.array([[-9, 1]])

    one_by_two_board_env = Environment(1, 2, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-10, goal_value=10)
    one_by_two_board_env.board = one_by_two_board # overwrite board
    one_by_two_board_env.dynamics = ONE_BY_TWO_DYNAMICS

    one_by_two_student_value_iteration = value_iteration_func(one_by_two_board_env, max_iterations=20_000, change_margin=1, gamma=0.9)
    assert np.allclose(one_by_two_student_value_iteration, one_by_two_board_value_iteration_solution, atol=2) == True, f"Test Value Iteration 1: Failed 1x2 Board. \nStudent Value Iteration {one_by_two_student_value_iteration} \nExpected Value Iteration {one_by_two_board_value_iteration_solution}"

    ########## Test 2x2 board ###############
    two_by_two_board = np.array([[-1, 5], [-5, -1]])
    two_by_two_board_value_iteration_solution = np.array([[14, 11],
       [11, 14]])

    two_by_two_env = Environment(2, 2, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    two_by_two_env.board = two_by_two_board  # overwrite board
    two_by_two_env.dynamics = TWO_BY_TWO_DYNAMICS

    two_by_two_student_value_iteration = value_iteration_func(two_by_two_env, max_iterations=20_000, change_margin=1, gamma=0.9)
    assert np.allclose(two_by_two_student_value_iteration, two_by_two_board_value_iteration_solution, atol=2) == True, f"Test Value Iteration 1: Failed 2x2 Board. \nStudent Value Iteration {two_by_two_student_value_iteration} \nExpected Value Iteration {two_by_two_board_value_iteration_solution}"
    
    ########## Test 5x5 board ###############
    five_by_five_board = np.array([
        [-1., -1., -1., -1., -1.],
        [-1., -5., -1., -1., -1.],
        [5., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1.]
        ])
    five_by_five_board_value_iteration_solution = np.array([
        [18.19212575, 15.34659918, 12.78886233, 10.47022654,  8.42428409],
        [21.31778853, 18.12807578, 15.16215726, 12.6218661 , 10.34694553],
        [18.18149787, 21.1827194 , 18.0256459 , 15.1470296 , 12.61202834],
        [21.29650849, 18.12121041, 15.23888889, 12.66873718, 10.38657086],
        [18.13575797, 15.32412045, 12.74910293, 10.46156843,  8.41163547]
    ])
    five_by_five_board_env =  Environment(5, 5, ActionMode.SIMPLE, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    five_by_five_board_env.board = five_by_five_board  # overwrite board
    five_by_five_board_env.dynamics = FIVE_BY_FIVE_DYNAMICS

    five_by_five_student_value_iteration = value_iteration_func(five_by_five_board_env, max_iterations=20_000, change_margin=1e-5, gamma=0.9)
    assert np.allclose(five_by_five_student_value_iteration,five_by_five_board_value_iteration_solution, atol=0.2) == True, f"Test Value Iteration 1: Failed 10x10 Board. \nStudent Value Iteration {five_by_five_student_value_iteration} \nExpected Value Iteration {five_by_five_board_value_iteration_solution}"
    
    print("Test Value Iteration 1 Passed")

def test_value_iteration_2(value_iteration_func):
    with open("dynamics.pkl", "rb") as f:
        data = pickle.load(f)
        CARDINAL_THREE_BY_THREE_DYNAMICS = data["CARDINAL_THREE_BY_THREE_DYNAMICS"]
        CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS = data["CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS"]
    ########## Test Cardinal Action Mode ###############
    cardinal_action_mode_solution = np.array([
        [21.22683963, 18.11729079, 21.26179756],
        [21.13562005, 21.16885773, 21.25114697],
        [17.97565107, 18.09694773, 18.06646864]])
    
    three_by_three_board = np.array([[-1, 5, -1.],
                                    [-1, -5, -1.],
                                    [-1, -1, -1.]])

    cardinal_action_mode_env = Environment(3, 3, ActionMode.CARDINAL, discrete_rewards=True, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    cardinal_action_mode_env.board = three_by_three_board
    cardinal_action_mode_env.dynamics = CARDINAL_THREE_BY_THREE_DYNAMICS
    
    cardinal_action_mode_student_value_iteration = value_iteration_func(cardinal_action_mode_env, max_iterations=20_000, change_margin=1e-5, gamma=0.9)
    assert np.allclose(cardinal_action_mode_student_value_iteration, cardinal_action_mode_solution, atol=0.2) == True, f"Test Value Iteration 2: Failed 3x3 Board Cardinal Action Mode. \nStudent Value Iteration {cardinal_action_mode_student_value_iteration} \nExpected Value Iteration {cardinal_action_mode_solution}"

    ########## Test Continuous Rewards ###############
    continuous_rewards_solution = np.array(
        [[6.66330363, 7.34513279, 7.0932941 , 6.91465241, 7.09480479],
       [7.35505612, 7.59426498, 7.35202349, 7.08229958, 6.9055746 ],
       [7.5981941 , 7.35938234, 7.58416933, 6.62716326, 6.24529379],
       [7.2735083 , 7.57923291, 7.28645234, 6.18251003, 6.50300099],
       [6.16014282, 7.29493055, 6.18628356, 6.51096226, 5.08298639]])
    five_by_five_board = np.array([[-0.8253895, -0.69805011, 0.25353471, 0.87788343, 0.53303922],
                                [ 0.04398602,  0.53420759, -0.50640458, -0.26112835,  0.03275385],
                                [-0.94955295,  0.98347051, -0.19441639,  0.15650246,  0.57653413],
                                [-0.38340712,  0.47741138, -0.36854612,  0.95404696, -0.76899452],
                                [-0.91146198, -0.56136624,  0.56397481, -0.94386589,  0.35540827]])

    continuous_rewards_env = Environment(5, 5, ActionMode.SIMPLE, discrete_rewards=False, obstacle_count=1, goal_count=1, obstacle_value=-5, goal_value=5)
    continuous_rewards_env.board = five_by_five_board
    continuous_rewards_env.dynamics = CONTINUOUTS_FIVE_BY_FIVE_DYNAMICS

    continuous_rewards_student_value_iteration = value_iteration_func(continuous_rewards_env, max_iterations=20_000, change_margin=1e-5, gamma=0.9)
    assert np.allclose(continuous_rewards_student_value_iteration, continuous_rewards_solution, atol=0.2) == True,  f"Test Value Iteration 2: Failed 5x5 Board Continuous Rewards. \nStudent Value Iteration {continuous_rewards_student_value_iteration} \nExpected Value Iteration {continuous_rewards_solution}"
    print("Test Value Iteration 2 Passed")

def test_policy_extraction_1(policy_extraction_func):
    ########## Test 1x2 board ###############
    one_by_two_board_value_function = np.array([[-9, 1]], dtype=np.float64)
    one_by_two_board_policy_extraction_solution = np.array([[[1], [3]]])

    one_by_two_student_policy_extraction = policy_extraction_func(one_by_two_board_value_function, ActionMode.SIMPLE)

    assert compare_policies(one_by_two_student_policy_extraction, one_by_two_board_policy_extraction_solution),\
        f"Test Policy Extraction 1: Failed 1x2 Board. \nStudent Policy Extraction:\n{one_by_two_student_policy_extraction} \nExpected Policy Extraction (possibilities):\n{one_by_two_board_policy_extraction_solution}"

    ########## Test 2x2 board ###############
    two_by_two_board_value_function = np.array([[13, 11], [10, 9]], dtype=np.float64)

    two_by_two_board_policy_extraction_solution = np.array([[[1], [3]], [[0], [0]]])

    two_by_two_student_policy_extraction = policy_extraction_func(two_by_two_board_value_function, ActionMode.SIMPLE)

    assert compare_policies(two_by_two_student_policy_extraction, two_by_two_board_policy_extraction_solution),\
            f"Test Policy Extraction 1: Failed 2x2 Board. \nStudent Policy Extraction:\n{two_by_two_student_policy_extraction} \nExpected Policy Extraction (possibilities):\n{two_by_two_board_policy_extraction_solution}"


    ########## Test 5x5 board ###############
    five_by_five_board_value_function = np.array([
        [18.14618091, 15.29833049, 12.76163595, 10.48405724, 8.43507376],
        [21.29817416, 18.07830039, 15.08061161, 12.57122232, 10.3033874],
        [18.16828939, 21.18659721, 17.9594393, 15.10623587, 12.56199901],
        [21.2869318, 18.06965591, 15.214279, 12.6734811, 10.36861585],
        [18.1494121, 15.31379495, 12.74895312, 10.44601046, 8.40123517]
    ], dtype=np.float64)
    five_by_five_board_policy_extraction_solution = np.array([
        [[2], [3], [3], [3], [3]],
        [[2], [3], [3], [2], [3]],
        [[0], [3], [3], [3], [3]],
        [[0], [3], [3], [3], [3]],
        [[0], [3], [3], [3], [3]]])

    five_by_five_student_policy_extraction = policy_extraction_func(five_by_five_board_value_function, ActionMode.SIMPLE)

    assert compare_policies(five_by_five_student_policy_extraction, five_by_five_board_policy_extraction_solution),\
        f"Test Policy Extraction 1: Failed 5x5 Board. \nStudent Policy Extraction:\n{five_by_five_student_policy_extraction} \nExpected Policy Extraction (possibilities):\n{five_by_five_board_policy_extraction_solution}"

    print("Test Policy Extraction 1 Passed")

def test_policy_extraction_2(policy_extraction_func):
    ########## Test Cardinal Action Mode ###############
    cardinal_action_mode_value_function = np.array([
        [21.09202072, 17.9746693, 21.1014615 ],
        [21.06235122, 21.01368932, 21.08269838],
        [17.94379359, 17.91411082, 17.94040373]])
    cardinal_action_mode_policy_extraction_solution = np.array([
        [[4], [2], [4]],
        [[0], [1], [0]],
        [[0], [1], [0]]])
    
    cardinal_action_mode_student_policy_extraction = policy_extraction_func(cardinal_action_mode_value_function, ActionMode.CARDINAL)

    assert compare_policies(cardinal_action_mode_student_policy_extraction, cardinal_action_mode_policy_extraction_solution),\
        f"Test Policy Extraction 2: Failed 3x3 Board Cardinal Action Mode. \nStudent Policy Extraction:\n{cardinal_action_mode_student_policy_extraction} \nExpected Policy Extraction (possibilities):\n{cardinal_action_mode_policy_extraction_solution}"
  
    ########## Test Continuous Rewards ###############
    continuous_rewards_value_function = np.array([
        [6.59144156, 7.28282042, 7.10626205, 6.92737036, 7.10643084],
        [7.27814262, 7.52093551, 7.28861742, 7.09467293, 6.92276667],
        [7.52420979, 7.29186867, 7.51724742, 6.56860971, 6.25992152],
        [7.214457,   7.51746602, 7.23156988, 6.12919053, 6.46007687],
        [6.10844948, 7.21547163, 6.13322726, 6.45569645, 5.04472821]])
    continuous_rewards_policy_extraction_solution = np.array([
        [[1], [2], [2], [1], [3]],
        [[2], [2], [3], [3], [0]],
        [[1], [3], [3], [3], [0]],
        [[0], [0], [3], [3], [0]],
        [[1], [0], [0], [3], [0]]])
    
    continuous_rewards_student_policy_extraction = policy_extraction_func(continuous_rewards_value_function, ActionMode.SIMPLE)
    
    assert compare_policies(continuous_rewards_student_policy_extraction, continuous_rewards_policy_extraction_solution),\
        f"Test Policy Extraction 2: Failed 5x5 Board Continuous Rewards. \nStudent Policy Extraction:\n{continuous_rewards_student_policy_extraction} \nExpected Policy Extraction (possibilities):\n{continuous_rewards_policy_extraction_solution}"
    
    print("Test Policy Extraction 2 Passed")

def test_policy_iteration_1(policy_iteration_func):
    np.random.seed(42)
    one_by_two_env, two_by_two_env, five_by_five_env = get_test_1_environments()

    ########## Test 1x2 board ###############
    one_by_two_value_function_solution = np.array([[-9, 1]], dtype=np.float64)
    one_by_two_policy_iteration_solution = np.array([[[1], [3]]])

    one_by_two_student_policy_iteration, one_by_two_student_value_function = policy_iteration_func(one_by_two_env, max_iterations=10, gamma=0.9, change_margin = 1)

    assert np.allclose(one_by_two_student_value_function, one_by_two_value_function_solution, atol=2) == True,\
        f"Test Policy Iteration 1: Failed 1x2 Board. \nStudent Value Function:\n{one_by_two_student_value_function} \nExpected Value Function:\n{one_by_two_value_function_solution}"
    assert compare_policies(one_by_two_student_policy_iteration, one_by_two_policy_iteration_solution),\
        f"Test Policy Iteration 1: Failed 1x2 Board. \nStudent Policy:\n{one_by_two_student_policy_iteration} \nExpected Policy (possibilities):\n{one_by_two_policy_iteration_solution}"

    ########## Test 2x2 board ###############
    two_by_two_value_function_solution = np.array([[133, 104], [104, 132]], dtype=np.float64)
    two_by_two_policy_iteration_solution = np.array([[[1, 2], [2, 2]], [[1, 1], [0, 3]]])
    two_by_two_student_policy_iteration, two_by_two_student_value_function = policy_iteration_func(two_by_two_env,  max_iterations=50, gamma=0.8, change_margin = 1)

    assert np.allclose(two_by_two_student_value_function, two_by_two_value_function_solution, atol=3) == True, f"Test Policy Iteration 1: Failed 2x2 Board. \nStudent Value Function:\n{two_by_two_student_value_function} \nExpected Value Function:\n{two_by_two_value_function_solution}"
    assert compare_policies(two_by_two_student_policy_iteration, two_by_two_policy_iteration_solution),\
        f"Test Policy Iteration 1: Failed 2x2 Board. \nStudent Policy:\n{two_by_two_student_policy_iteration} \nExpected Policy (possibilities):\n{two_by_two_policy_iteration_solution}"

    ########## Test 5x5 board ###############
    five_by_five_value_function_solution = np.array([
        [[198.11164798, 195.08167652, 192.06790888, 189.03809398, 186.17924609],
       [201.03012035, 197.89859596, 194.80786819, 191.87103078, 189.00239601],
       [198.07844066, 200.8287113 , 197.84121431, 194.79555384, 191.84527217],
       [201.02679582, 197.98386908, 194.94688478, 191.98965228, 189.07901317],
       [197.99837218, 195.05955027, 192.03803723, 189.07364104, 186.14350192]]
    ], dtype=np.float64)
    five_by_five_policy_iteration_solution_possibilities = np.array([
        [[2, 2, 2], [2, 3, 3], [2, 3, 3], [2, 3, 3], [2, 3, 3]],
        [[2, 2, 2], [2, 3, 3], [2, 3, 3], [2, 3, 3], [2, 3, 3]],
        [[0, 1, 2], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[0, 0, 0], [0, 3, 3], [0, 3, 3], [0, 3, 3], [0, 3, 3]],
        [[0, 0, 0], [0, 3, 3], [0, 3, 3], [0, 3, 3], [0, 3, 3]],
    ])

    five_by_five_student_policy_iteration, five_by_five_student_value_function =  policy_iteration_func(five_by_five_env, max_iterations=500, gamma=0.99, change_margin = 1e-5)
    assert np.allclose(five_by_five_student_value_function,five_by_five_value_function_solution, atol=0.1) == True,\
        f"Test Policy Iteration 1: Failed 5x5 Board. \nStudent Value Function:\n{five_by_five_student_value_function} \nExpected Value Function:\n{five_by_five_value_function_solution}"
    assert compare_policies(five_by_five_student_policy_iteration, five_by_five_policy_iteration_solution_possibilities),\
        f"Test Policy Iteration 1: Failed 5x5 Board. \nStudent Policy:\n{five_by_five_student_policy_iteration} \nExpected Policy (possibilities):\n{five_by_five_policy_iteration_solution_possibilities}\n"

    print("Test Policy Iteration 1 Passed")

def test_policy_iteration_2(policy_iteration_func):
    np.random.seed(42)
    cardinal_action_mode_env, continuous_rewards_env = get_test_2_environments()

    ########## Test Cardinal Action Mode ###############
    cardinal_action_mode_value_function_solution = np.array(
        [[200.74696989, 197.85242741, 200.80441932],
       [200.74066366, 200.76553887, 200.81045184],
       [197.60026273, 197.84686938, 197.73687383]]
    )
    
    cardinal_action_mode_policy_iteration_solution = np.array([
        [[2, 2, 2, 2], [2, 3, 5, 6], [6, 6, 6, 6]],
        [[1, 1, 1, 1], [0, 0, 0, 0], [7, 7, 7, 7]],
        [[0, 1, 0, 1], [0, 1, 7, 7], [0, 0, 0, 7]]])
    
    cardinal_action_mode_student_policy_iteration, cardinal_action_mode_student_value_function = policy_iteration_func(cardinal_action_mode_env,  max_iterations=500, gamma=0.99, change_margin = 1e-5)

    assert np.allclose(cardinal_action_mode_student_value_function, cardinal_action_mode_value_function_solution, atol=0.1) == True, f"Test Policy Iteration 2: Failed 3x3 Board Cardinal Action Mode. \nStudent Value Function:\n{cardinal_action_mode_student_value_function} \nExpected Value Function:\n{cardinal_action_mode_value_function_solution}"
    assert compare_policies(cardinal_action_mode_student_policy_iteration, cardinal_action_mode_policy_iteration_solution), \
        f"Test Policy Iteration 2: Failed 3x3 Board Cardinal Action Mode. \nStudent Policy:\n{cardinal_action_mode_student_policy_iteration} \nExpected Policy (possibilities)::\n{cardinal_action_mode_policy_iteration_solution}"

    ########## Test Continuous Rewards ###############
    continuous_rewards_value_function_solution = np.array(
       [[71.91079529, 72.56176905, 71.30041445, 70.83814189, 70.98267414],
       [72.56197266, 72.78419293, 72.55411822, 71.3707991 , 71.20677148],
       [72.84433722, 72.57096409, 72.81705449, 71.9116406 , 71.3331337 ],
       [72.53498056, 72.78195244, 72.54338235, 71.44227594, 71.65615525],
       [71.41226701, 72.5283285 , 71.44669486, 71.67278855, 70.16926544]])
    continuous_rewards_policy_iteration_solution = np.array([
        [[2, 2], [2, 2], [2, 2], [3, 3], [3, 3]],
        [[1, 1], [2, 2], [3, 3], [2, 2], [2, 2]],
        [[1, 1], [0, 2], [3, 3], [3, 3], [3, 3]],
        [[1, 1], [0, 0], [3, 3], [0, 3], [3, 3]],
        [[0, 0], [0, 0], [0, 3], [0, 0], [0, 0]]])
    
    continuous_rewards_student_policy_iteration, continuous_rewards_student_value_function = policy_iteration_func(continuous_rewards_env,  max_iterations=600, gamma=0.99, change_margin = 1e-5)

    assert np.allclose(continuous_rewards_student_value_function, continuous_rewards_value_function_solution, atol=0.1) == True,  f"Test Policy Iteration 2: Failed 5x5 Board Continuous Rewards. \nStudent Value Function:\n{continuous_rewards_student_value_function} \nExpected Value Function:\n{continuous_rewards_value_function_solution}"
    assert compare_policies(continuous_rewards_student_policy_iteration, continuous_rewards_policy_iteration_solution),\
        f"Test Policy Iteration 2: Failed 5x5 Board Continuous Rewards. \nStudent Policy:\n{continuous_rewards_student_policy_iteration} \nExpected Policy (possibilities):\n{continuous_rewards_policy_iteration_solution}"

    print("Test Policy Iteration 2 Passed")
