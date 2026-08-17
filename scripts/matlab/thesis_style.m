function thesis_style(ax)
%THESIS_STYLE  Apply the thesis figure styling to one axes object.
%
%   THESIS_STYLE(AX) sets the fonts, grid, box and tick behaviour used by
%   every figure in the main body, so that figures produced in separate
%   scripts are visually indistinguishable from one another.
%
%   The settings match the LaTeX document: 11 pt Arial body text, with figure
%   text set two points smaller so that a figure scaled to the text width
%   carries labels of approximately body size. Exporting at these settings and
%   then rescaling in LaTeX defeats the purpose; export at the final width.
%
%   Inputs
%   ------
%   ax : matlab.graphics.axis.Axes
%       Axes to style. Defaults to GCA.
%
%   See also LOAD_THESIS_DATA, EXPORTGRAPHICS.

    if nargin < 1 || isempty(ax)
        ax = gca;
    end

    set(ax, ...
        'FontName',      'Arial', ...
        'FontSize',      9, ...
        'Box',           'on', ...
        'LineWidth',     0.75, ...
        'TickDir',       'in', ...
        'TickLength',    [0.012 0.012], ...
        'XMinorTick',    'on', ...
        'YMinorTick',    'on', ...
        'GridAlpha',     0.15, ...
        'MinorGridAlpha',0.08, ...
        'Layer',         'top');

    grid(ax, 'on');

    % LaTeX interpreter throughout, so that the mathematics in the axis labels
    % is set by the same engine as the mathematics in the body text.
    set(ax, 'TickLabelInterpreter', 'latex');
    if ~isempty(ax.XLabel), ax.XLabel.Interpreter = 'latex'; end
    if ~isempty(ax.YLabel), ax.YLabel.Interpreter = 'latex'; end
    if ~isempty(ax.Title),  ax.Title.Interpreter  = 'latex'; end

    lg = findobj(ax.Parent, 'Type', 'Legend');
    for k = 1:numel(lg)
        lg(k).Interpreter = 'latex';
        lg(k).FontSize    = 8;
        lg(k).Box         = 'off';
    end
end
