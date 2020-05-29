import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_sequence(params):

    taskselect = params.task

    sequence = []

    shapes = {
        0: 'circle',
        1: 'triangle',
        2: 'square'
    }

    nbr_of_trials = params.trials
    bag_size = 3
    nbr_of_bags = int(nbr_of_trials / bag_size)

    # bag of bags version; shape can repeat maximum 6 times in a row
    if taskselect == 'b0b':

        nbr_of_bobs = int(nbr_of_trials / (bag_size * len(shapes)))
        bag_o_bag = shapes

        for b0b in range(nbr_of_bobs):
            current_bob = list(bag_o_bag.keys())

            while len(current_bob) > 0:
                bag = np.random.choice(current_bob)

                for trial in range(bag_size):
                    sequence.append(shapes[bag])

                current_bob.pop(current_bob.index(bag))

    # random version; shape repeats minimum 3 times and no theoretical max
    if taskselect == 'random':

        for _ in range(nbr_of_bags):
            bag = np.random.choice(list(shapes.keys()))

            for trial in range(bag_size):
                sequence.append(shapes[bag])

    # bag of bags no repeat; shape can repeat maximum 3 times
    if taskselect == 'b0b-nr':

        bag_o_bag = shapes

        nbr_of_bobs = int(nbr_of_trials / (bag_size * len(shapes)))

        last_bag = -1
        for b0b in range(nbr_of_bobs):
            current_bob = list(bag_o_bag.keys())

            while len(current_bob) > 0:
                bag = np.random.choice(current_bob)

                while bag == last_bag:
                    bag = np.random.choice(current_bob)

                for trial in range(bag_size):
                    sequence.append(shapes[bag])

                last_bag = current_bob.pop(current_bob.index(bag))

    return sequence


def print_debug(observation, prediction, action, next_observation, qvalues):
    print(f'we observe {observation}')
    print(f'this gives us the predictions: {prediction}')
    print(f'we took action {action}')
    print(f'next observation will be {next_observation}')
    print(f'weight matrix after update:\n {qvalues}')


def simulate_subject(params, shape_sequence, debug=False):
    """
        simulates a single subject using standard Q-learning (QL)
        in cogsci paper we all it BQL - basic Q learning
        note that it doesnt use rewards, only predictions
    """

    # recast human concepts to machine language
    representation_map = {
        'circle':   [1, 0, 0],
        'triangle': [0, 1, 0],
        'square':   [0, 0, 1]
    }
    stimulus_sequence = [representation_map[x] for x in shape_sequence]

    # initialize qvalues matrix and chosen actions array
    qvalues = np.random.rand(3, 3)
    actions = np.zeros(params.trials)

    for trial, observation in enumerate(stimulus_sequence):

        # matrix multiplication will get q values for current observation (state)
        prediction = np.dot(observation, qvalues)

        # calculate action probabilities with softmax
        exp_qvals = np.exp(params.beta * prediction)
        probs = exp_qvals / np.sum(exp_qvals)

        # select action based on probabilities and save it
        action = np.random.choice([0, 1, 2], p=probs)
        actions[trial] = action

        # can't get next observation if we're on last trial
        if trial == len(stimulus_sequence) - 1:
            break

        next_observation = stimulus_sequence[trial + 1]
        next_prediction = np.dot(next_observation, qvalues)
        next_max_value = np.max(next_prediction)
        next_max_index = np.argmax(next_prediction)
        maxQ = np.zeros(3)
        maxQ[next_max_index] = next_max_value

        prediction_error = next_observation + params.gamma*maxQ - prediction

        state = np.argmax(observation)  # because row/column equivalency
        qvalues[state][action] += params.alpha * prediction_error[action]

        if debug:
            print_debug(observation,
                        prediction,
                        action,
                        next_observation,
                        qvalues)

    return actions, stimulus_sequence


def create_dataframe(actions, shape_sequence):
    """
        actions: 1D array of integers indication action choices
        shape_sequence: sequence of shapes in array form;
            [1, 0, 0] for circle,
            [0, 1, 0] for triangle,
            [0, 0, 1] for square
    """

    # map shape sequence and actions to string descriptions
    df = pd.DataFrame()
    df['stimulus'] = shape_sequence
    df['stimulus'] = df['stimulus'].apply(str)  # trick so we can map
    df['stimulus'] = df['stimulus'].map({
        '[1, 0, 0]': 'circle',
        '[0, 1, 0]': 'triangle',
        '[0, 0, 1]': 'square'
    })
    df['pred_choice'] = pd.Series(actions)
    df['pred_shape'] = df['pred_choice'].map({
        0: 'circle',
        1: 'triangle',
        2: 'square'
    })

    # create column showing if prediction is correct
    # shift pred_shape down one step, check if same as next ballcolor
    # shift back up to record if prediction in current trial is correct
    temp = df['stimulus'] == df['pred_shape'].shift(1)
    df['pred_correct'] = temp.shift(-1)

    # give a point if prediction differs from what is currently seen on screen
    # this should provide us nicely that 3rd shape in standard graphs differs
    shift_predict = df['pred_shape'] != df['stimulus']
    df['shift_predict'] = shift_predict

    # if prediction on this trial is same as prediction on last trial and
    # last trial was correct = WIN STAY
    last_same_this_trial = df['pred_shape'] == df['pred_shape'].shift(1)
    last_correct = df['pred_correct'].shift(1)
    winstay = last_same_this_trial + last_correct
    df['winstay'] = winstay == 2
    winstay_index = df.columns.get_loc('winstay')
    df.iloc[0, winstay_index] = np.nan

    # if prediction on this trial is diff from prediction on last trial and
    # last trial was incorrect = LOSE SHIFT
    last_diff_this_trial = df['pred_shape'] != df['pred_shape'].shift(1)
    last_incorrect = [0 if x else 1 for x in df['pred_correct'].shift(1)]
    lose_shift = last_diff_this_trial + last_incorrect
    df['loseshift'] = lose_shift == 2
    loseshift_index = df.columns.get_loc('loseshift')
    df.iloc[0, loseshift_index] = np.nan

    # final sanity check that no row is both winstay and loseshift
    check = (df['winstay'] + df['loseshift']) > 1
    assert( sum(check) == 0), "one or more trials is both winstay and loseshift"

    return df


def reshape_data(subject_df, params):
    """
        this prepares data for easier plotting.

        TODO: improve docstring
    """

    nbr_of_bags = int(params.trials / 3)
    nbr_of_bobs = int(nbr_of_bags / 3)
    shape_pos = np.tile([1,2,3], nbr_of_bags)
    bins = np.array([], dtype='int16')
    bins = np.array([np.append(bins, np.repeat(x, nbr_of_bobs)) for x in range(1, 10)])
    bins = bins.flatten()

    pred_correct = pd.DataFrame()
    winstay = pd.DataFrame()
    loseshift = pd.DataFrame()
    shiftpredict = pd.DataFrame()

    pred_correct['mysteryperson'] = subject_df.pred_correct
    winstay['mysteryperson'] = subject_df.winstay
    loseshift['mysteryperson'] = subject_df.loseshift
    shiftpredict['mysteryperson'] = subject_df.shift_predict

    pred_correct['shape_pos'] = shape_pos
    winstay['shape_pos'] = shape_pos
    loseshift['shape_pos'] = shape_pos
    shiftpredict['shape_pos'] = shape_pos

    # bob might only be appropriate for 99 trials could also call it bin
    pred_correct['bob'] = bins
    winstay['bob'] = bins
    loseshift['bob'] = bins
    shiftpredict['bob'] = bins

    df_dict = {
        'pred_correct': pred_correct,
        'winstay': winstay,
        'loseshift': loseshift,
        'shiftpredict': shiftpredict
    }

    return df_dict


def barplot_shape(shape_pos, dataframes, params, save = False):
    """
        plot barchart for shape at position `shape_pos`
        TODO: improve this message and maybe function as well
        dataframes is dict object for now not ideal i think
    """

    df = pd.DataFrame()

    df['accuracy'] = dataframes['pred_correct'].query('shape_pos == @shape_pos').drop(columns=['shape_pos', 'bob']).mean()
    df['winstay'] = dataframes['winstay'].query('shape_pos == @shape_pos').drop(columns=['shape_pos', 'bob']).mean()
    df['loseshift'] = dataframes['loseshift'].query('shape_pos == @shape_pos').drop(columns=['shape_pos', 'bob']).mean()
    df['shiftpredict'] = dataframes['shiftpredict'].query('shape_pos == @shape_pos').drop(columns=['shape_pos', 'bob']).mean()

    plt.figure(figsize=(20,10), dpi=200)
    plt.rc('xtick',labelsize=30)
    plt.rc('ytick',labelsize=20)
    sns.barplot(x='variable', y='value', data=df.melt())
    plt.title(f'S{shape_pos} for {params}', fontsize=30)  # TODO! change
    plt.xlabel('scoring', fontsize=20)
    plt.ylabel('probability', fontsize=30)
    plt.ylim(0, 1)
    if save:
        # TODO! change
        plt.savefig(f'SOMEPATH/yeahman_ball{shape_pos}.png', dpi=300)
    plt.show()


def prep_data_for_choiceplot(shape_array, yval = 1):
    """
        input an array/sequence of shapes and get back a melted dataframe where each symbol has value yval so they can be plotted on same line
    """

    df = pd.DataFrame()
    df['symbol'] = shape_array

    df['circle'] = df.symbol.apply(lambda x: yval if x == 'circle' else -2)
    df['square'] = df.symbol.apply(lambda x: yval if x == 'square' else -2)
    df['triangle'] = df.symbol.apply(lambda x: yval if x == 'triangle' else -2)

    df2 = df.drop(columns=['symbol']).reset_index().melt('index', var_name='shape', value_name='value')

    return df2


def choiceplot(actions, observations):

    plt.figure(figsize=(25, 4), dpi=100)
    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    # marker list https://matplotlib.org/3.1.0/api/markers_api.html
    sns.scatterplot(x='index', y='value', hue='shape', data=observations, s=150, markers={'circle': 'o', 'triangle': '^', 'square': 's'}, style='shape', legend=False)
    sns.scatterplot(x='index', y='value', hue='shape', data=actions, s=150, markers={'circle': 'o', 'triangle': '^', 'square': 's'}, style='shape', legend=False)
    plt.ylim(0, 3)
    plt.show()


def lineplot(df_dict, params):
    """
        input: df_dict from reshape_data
        output: a nice lineplot of the four measures accuracy, winstay, loseshift and shiftpredict!
    """

    # calculate ratio correct over bag-o-bags / bin
    # drop last trial as it cannot have correct prediction
    pred_correct = df_dict['pred_correct'].drop([98]).drop(columns=['shape_pos']).groupby('bob').agg(lambda x: sum(x) / len(x)).reset_index(drop=True)
    pred_correct = pred_correct.rename(columns={'mysteryperson': 'pred_correct'})

    # drop first trial as it cant have winstay point
    winstay = df_dict['winstay'].drop([0]).drop(columns=['shape_pos']).groupby('bob').agg(lambda x: sum(x) / len(x)).reset_index(drop=True)
    winstay = winstay.rename(columns={'mysteryperson': 'winstay'})

    # drop first trial as it cant have loseshift
    loseshift = df_dict['loseshift'].drop([0]).drop(columns=['shape_pos']).groupby('bob').agg(lambda x: sum(x) / len(x)).reset_index(drop=True)
    loseshift = loseshift.rename(columns={'mysteryperson': 'loseshift'})

    # shift predict can be true any time so no need to drop trials
    shiftpredict = df_dict['shiftpredict'].drop(columns=['shape_pos']).groupby('bob').agg(lambda x: sum(x) / len(x)).reset_index(drop=True)
    shiftpredict = shiftpredict.rename(columns={'mysteryperson': 'shiftpredict'})

    # join all the measures together
    combined = shiftpredict.join(loseshift).join(winstay).join(pred_correct).reset_index().melt('index')

    # this with hues when we have all 4
    plt.figure(figsize=(20, 5), dpi=200)
    sns.lineplot(x='index', y='value', hue='variable', data=combined, lw=3)
    plt.title(f'{params}', fontsize=20)
