import matplotlib.pyplot as plt
import numpy as np

def scatter_choices(individual_utilities, social_utilities, current_choice, 
        size, transparency, no_transparency, numbers, label, ax, color=[1,1,1],
        marker='o'):
    """Draw the choices of an individual

    :individual_utilities: list with the individual utility for each choice
    :social_utilities: list with the social utility for each choice
    :current_choice: index of the current choice of the individual
    :color: list with rgb values for the color of the dots
    :size: size of the dots
    :transparency: percentage of transparency for dots of choices with lower utility than the current choice
    :no_transparency: percentage of transparency for dots of choices with higer or equal utility than the current choice
    :numbers: if true, numbers are added next to the dots
    :label: label to add in the legend
    :ax: ax object
    :returns: add a scatter object to the plot

    """
    J = len(individual_utilities)
    rgba_colors = np.zeros((J, 4))
    rgba_colors[:, 0:3] = color
    alphas = np.append(
            [transparency for i in range(current_choice)], 
            [no_transparency for i in range(J - current_choice)]
    )
    rgba_colors[:, 3] = alphas
    edge_colors = np.ones((J, 4))
    edge_colors[current_choice, 0:3] = 0
    ax.scatter(
            individual_utilities, social_utilities, s=size, color=rgba_colors, 
            edgecolor=edge_colors, label=label, marker=marker
    )
    if numbers == True:
        for i, number in enumerate(np.arange(0, J, 1)):
            ax.annotate(number+1, (individual_utilities[i]+.1, social_utilities[i]+.1))
    pass

def efficiency_arrow(individual_utilities, social_utilities, current_choice,
        next_choice, ax, additional_text=''):
    """Draw an arrow going from choice j to choice j+1

    :individual_utilities: list with the individual utility for each choice
    :social_utilities: list with the social utility for each choice
    :current_choice: index of the current choice of the individual
    :next_choice: index of the target choice
    :ax: ax object
    :returns: add an arrow to the plot

    """
    efficiency = - ( social_utilities[next_choice] - social_utilities[current_choice] ) / ( individual_utilities[next_choice] - individual_utilities[current_choice] )
    if efficiency.is_integer() == True:
        efficiency = int(efficiency)
    x = ( individual_utilities[next_choice] + individual_utilities[current_choice] ) / 2
    y = ( social_utilities[next_choice] + social_utilities[current_choice] ) / 2
    ax.annotate(str(efficiency)+additional_text, (x+.02, y+.02))
    ax.annotate("",
            xy = (individual_utilities[current_choice], social_utilities[current_choice]),
            xytext = (individual_utilities[next_choice], social_utilities[next_choice]),
            arrowprops = dict(arrowstyle = "<-", connectionstyle = "arc3", shrinkA = 6, shrinkB = 6),
            )
    pass

x3 = [4, 3, 2, 1]
x2 = [6, 5, 4, 3]
x1 = [8, 7, 6, 5]
y1 = [0, 1, 7, 10]
y2 = [0, 5, 6, 10]
y3 = [0, 4, 6, 7]
x1i = 0
x2i = 0
x3i = 0
x1j = 2
x2j = 1
x3j = 1

fig, ax = plt.subplots()

scatter_choices(
        individual_utilities=x3, social_utilities=y3, current_choice=x3i, 
        color=[.2, .2, .8], 
        #marker=(6,2,0),
        size=100, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual 1', 
        ax=ax
)
scatter_choices(
        individual_utilities=x2, social_utilities=y2, current_choice=x2i, 
        color=[.2, .8, .2], 
        marker=(6,1,0),
        size=200, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual 2', 
        ax=ax
)
scatter_choices(
        individual_utilities=x1, social_utilities=y1, current_choice=x1i, 
        color=[.8, .2, .2], 
        marker=(4,0,0),
        size=150, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual 3', 
        ax=ax
)

efficiency_arrow(individual_utilities=x1, social_utilities=y1, current_choice=0,
        next_choice=2, ax=ax, additional_text=' (III)')
efficiency_arrow(individual_utilities=x1, social_utilities=y1, current_choice=2,
        next_choice=3, ax=ax, additional_text=' (IV)')
efficiency_arrow(individual_utilities=x2, social_utilities=y2, current_choice=0,
        next_choice=1, ax=ax, additional_text=' (I)')
efficiency_arrow(individual_utilities=x2, social_utilities=y2, current_choice=1,
        next_choice=3, ax=ax, additional_text=' (V)')
efficiency_arrow(individual_utilities=x3, social_utilities=y3, current_choice=0,
        next_choice=1, ax=ax, additional_text=' (II)')
#efficiency_arrow(individual_utilities=x3, social_utilities=y3, current_choice=1,
        #next_choice=2, ax=ax, additional_text=' (VI)')
#efficiency_arrow(individual_utilities=x3, social_utilities=y3, current_choice=2,
        #next_choice=3, ax=ax, additional_text=' (VII)')

ax.legend()

ax.set_title('')
ax.set_xlabel('Individual utility')
ax.set_ylabel('Social utility')

plt.savefig('algorithm-schema.png', dpi=300, format='png')
