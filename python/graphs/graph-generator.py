import matplotlib.pyplot as plt
import numpy as np

# Define colors for the graphs.
color1 = '#608f42'
color2 = '#90ced6'
color3 = '#54251e'

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    rgb = list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    rgb = [x/255 for x in rgb]
    return rgb

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
    if isinstance(color, str):
        rgba_colors = hex_to_rgb(color)
    else:
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
        next_choice, ax, additional_text='', thick=False):
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
    else:
        efficiency = round(efficiency, 2)
    x = ( 2 * individual_utilities[next_choice] + individual_utilities[current_choice] )/3
    y = ( 2 * social_utilities[next_choice] + social_utilities[current_choice] )/3
    ax.annotate(additional_text + ' (' + str(efficiency) + ')', (x, y))
    if thick:
        l = 2
    else:
        l = 1
    ax.annotate("",
            xy = (individual_utilities[current_choice], social_utilities[current_choice]),
            xytext = (individual_utilities[next_choice], social_utilities[next_choice]),
            arrowprops = dict(arrowstyle = "<-", connectionstyle = "arc3",
                shrinkA = 6, shrinkB = 6, linewidth=l),
            )
    pass

x3 = [4, 3, 2, 1]
x2 = [6, 5, 4, 3]
x1 = [8, 7, 6, 5]
y1 = [1, 2, 8, 11]
y2 = [1, 6, 7, 11]
y3 = [1, 5, 7, 8]
x1i = 0
x2i = 0
x3i = 0
x1j = 2
x2j = 1
x3j = 1

fig, ax = plt.subplots()

scatter_choices(
        individual_utilities=x3, social_utilities=y3, current_choice=x3i, 
        color=color1,
        #marker=(6,2,0),
        size=150, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual A', 
        ax=ax
)
scatter_choices(
        individual_utilities=x2, social_utilities=y2, current_choice=x2i, 
        color=color2,
        #marker=(6,1,0),
        size=150, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual B', 
        ax=ax
)
scatter_choices(
        individual_utilities=x1, social_utilities=y1, current_choice=x1i, 
        color=color3,
        #marker=(4,0,0),
        size=150, transparency=.25, no_transparency=.75, 
        numbers=True, 
        label='Individual C', 
        ax=ax
)

if True:
    efficiency_arrow(individual_utilities=x2, social_utilities=y2, current_choice=0,
            next_choice=1, ax=ax, additional_text=' I', thick=True)
    efficiency_arrow(individual_utilities=x3, social_utilities=y3, current_choice=0,
            next_choice=1, ax=ax, additional_text=' II', thick=True)
    efficiency_arrow(individual_utilities=x1, social_utilities=y1, current_choice=0,
            next_choice=2, ax=ax, additional_text=' III', thick=True)
    efficiency_arrow(individual_utilities=x1, social_utilities=y1, current_choice=2,
            next_choice=3, ax=ax, additional_text='')
    efficiency_arrow(individual_utilities=x2, social_utilities=y2, current_choice=1,
            next_choice=3, ax=ax, additional_text='')
    efficiency_arrow(individual_utilities=x3, social_utilities=y3, current_choice=1,
            next_choice=2, ax=ax, additional_text='')

ax.legend()

ax.set_title('')
ax.set_xlabel('Individual utility')
ax.set_ylabel('Social utility')

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.savefig('algorithm-schema.pdf', dpi=600, format='pdf')
