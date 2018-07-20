%function Plot_spl(X, y, y_pred, X0, y0, errb, errb0)
function Plot_spl(X, y, X0, y0, errb0, color)

  figure

    %plot(X, y_pred, 'red')   % predict on training x
    %hold on
    %plot(X, y_pred + errb, ':red', X, y_pred - errb, ':red')   % error band
    %hold on
    %plot(X0, y0 + errb0, ':blue', X0, y0 - errb0, ':blue')   % error band
    %hold on
    fill([X0; flipud(X0)], [y0 - errb0; flipud(y0 + errb0)], 'yellow', 'linestyle','none')
    hold on
    scatter(X, y, 20, 'filled')   % original training data
    hold on
    %plot(X0, y0, 'blue')
    plot(X0, y0, color)

    xlabel('age')
    ylabel('BMD')

end

