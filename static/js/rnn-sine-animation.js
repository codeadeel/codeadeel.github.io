(function () {
  "use strict";

  if (window.__rnnSineAnimationLoaded) return;
  window.__rnnSineAnimationLoaded = true;

  const B = 5;
  const T = 10;
  const F = 1;
  const H = 4;
  const C = 2;
  const OPERATIONS_PER_TIME_STEP = 4;
  const FINAL_POSITION = T * OPERATIONS_PER_TIME_STEP;
  const STAGE_NAMES = [
    "Slice x_t",
    "Input × W_x",
    "Memory × W_h",
    "Add + tanh",
  ];
  const SIGNAL_NAMES = Array.from({ length: B }, function (_, index) {
    return "S" + index;
  });
  const HIDDEN_NAMES = Array.from({ length: H }, function (_, index) {
    return "h" + index;
  });

  // These are fixed weights from a tiny trained PyTorch RNN. Keeping them
  // fixed makes this browser visualization deterministic and fast.
  const MODEL = {
    weightX: [[1.118655, 0.566091, -0.464106, -0.729741]], // (1,4)
    weightH: [
      [0.453605, -0.685469, -0.813263, -0.608306],
      [0.153480, 0.247879, -0.750757, 1.026104],
      [0.234076, -1.031029, 1.124806, -0.465925],
      [-0.618223, 0.590524, -0.926612, 0.865281],
    ], // (4,4)
    bias: [0.116025, -0.438044, -0.156052, 0.24656], // (4,)
    classifierWeight: [
      [-0.097417, -1.466285, 1.219051, -0.480312],
      [-0.1349, 1.09015, -1.352753, 0.766938],
    ], // stored by PyTorch as (2,4)
    classifierBias: [0.340712, -0.124128], // (2,)
  };

  const LABELS = [0, 1, 0, 1, 1];
  const PHASES = [0.0, 0.4, 1.0, 1.6, 2.2];

  function makeSignals() {
    return LABELS.map(function (label, row) {
      const cycles = label === 0 ? 1 : 3;
      return Array.from({ length: T }, function (_, timeStep) {
        return Math.sin(
          2 * Math.PI * cycles * (timeStep / T) + PHASES[row]
        );
      });
    });
  }

  function zeros(rows, columns) {
    return Array.from({ length: rows }, function () {
      return Array(columns).fill(0);
    });
  }

  function transpose(matrix) {
    return matrix[0].map(function (_, column) {
      return matrix.map(function (row) {
        return row[column];
      });
    });
  }

  function matMul(left, right) {
    const rows = left.length;
    const shared = left[0].length;
    const columns = right[0].length;
    const output = zeros(rows, columns);

    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        let value = 0;
        for (let inner = 0; inner < shared; inner += 1) {
          value += left[row][inner] * right[inner][column];
        }
        output[row][column] = value;
      }
    }
    return output;
  }

  function addMatricesAndBias(first, second, bias) {
    return first.map(function (row, rowIndex) {
      return row.map(function (value, column) {
        return value + second[rowIndex][column] + bias[column];
      });
    });
  }

  function tanhMatrix(matrix) {
    return matrix.map(function (row) {
      return row.map(Math.tanh);
    });
  }

  function addBias(matrix, bias) {
    return matrix.map(function (row) {
      return row.map(function (value, column) {
        return value + bias[column];
      });
    });
  }

  function softmaxRows(matrix) {
    return matrix.map(function (row) {
      const maximum = Math.max.apply(null, row);
      const exponentials = row.map(function (value) {
        return Math.exp(value - maximum);
      });
      const denominator = exponentials.reduce(function (sum, value) {
        return sum + value;
      }, 0);
      return exponentials.map(function (value) {
        return value / denominator;
      });
    });
  }

  function argmax(row) {
    return row.reduce(function (best, value, index, values) {
      return value > values[best] ? index : best;
    }, 0);
  }

  function computeWalkthrough(signals) {
    const records = [];
    let hidden = zeros(B, H);

    for (let timeStep = 0; timeStep < T; timeStep += 1) {
      const xT = signals.map(function (signal) {
        return [signal[timeStep]];
      }); // (5,1)

      const inputProduct = matMul(xT, MODEL.weightX); // (5,4)
      const memoryProduct = matMul(hidden, MODEL.weightH); // (5,4)
      const preActivation = addMatricesAndBias(
        inputProduct,
        memoryProduct,
        MODEL.bias
      ); // (5,4)
      const newHidden = tanhMatrix(preActivation); // (5,4)

      records.push({
        timeStep: timeStep,
        xT: xT,
        hiddenPrevious: hidden,
        inputProduct: inputProduct,
        memoryProduct: memoryProduct,
        preActivation: preActivation,
        newHidden: newHidden,
      });

      hidden = newHidden;
    }

    const classifierWeightT = transpose(MODEL.classifierWeight); // (4,2)
    const logits = addBias(
      matMul(hidden, classifierWeightT),
      MODEL.classifierBias
    ); // (5,2)
    const probabilities = softmaxRows(logits); // (5,2)
    const predictions = probabilities.map(argmax); // (5,)

    let correct = 0;
    let loss = 0;
    const confusion = [
      [0, 0],
      [0, 0],
    ];

    LABELS.forEach(function (label, index) {
      const prediction = predictions[index];
      if (prediction === label) correct += 1;
      confusion[label][prediction] += 1;
      loss -= Math.log(Math.max(probabilities[index][label], 1e-12));
    });

    return {
      records: records,
      finalHidden: hidden,
      classifierWeightT: classifierWeightT,
      logits: logits,
      probabilities: probabilities,
      predictions: predictions,
      accuracy: correct / B,
      loss: loss / B,
      confusion: confusion,
    };
  }

  function formatNumber(value) {
    if (Math.abs(value) < 0.005) return "0.00";
    return value.toFixed(2);
  }

  function formatPrecise(value) {
    if (Math.abs(value) < 0.00005) return "0.0000";
    return value.toFixed(4);
  }

  function matrixShape(matrix) {
    return "(" + matrix.length + "," + matrix[0].length + ")";
  }

  function cellColor(value, matrix) {
    let maximum = 0;
    matrix.forEach(function (row) {
      row.forEach(function (candidate) {
        maximum = Math.max(maximum, Math.abs(candidate));
      });
    });
    const strength = maximum === 0 ? 0 : Math.min(Math.abs(value) / maximum, 1);
    const alpha = 0.06 + 0.26 * strength;
    return value >= 0
      ? "rgba(37, 99, 235, " + alpha.toFixed(3) + ")"
      : "rgba(214, 69, 69, " + alpha.toFixed(3) + ")";
  }

  function matrixTable(matrix, options) {
    const settings = options || {};
    const columnLabels = settings.columnLabels || [];
    const rowLabels = settings.rowLabels || [];
    const selectedRow = settings.selectedRow;
    const selectedColumn = settings.selectedColumn;

    const header =
      columnLabels.length > 0
        ? "<thead><tr>" +
          (rowLabels.length > 0 ? '<th scope="col"></th>' : "") +
          columnLabels
            .map(function (label, column) {
              return (
                '<th scope="col"' +
                (column === selectedColumn
                  ? ' class="is-selected-column"'
                  : "") +
                ">" +
                label +
                "</th>"
              );
            })
            .join("") +
          "</tr></thead>"
        : "";

    const rows = matrix
      .map(function (row, rowIndex) {
        const cells = row
          .map(function (value, column) {
            return (
              '<td title="' +
              value.toFixed(6) +
              '"' +
              (column === selectedColumn
                ? ' class="is-selected-column"'
                : "") +
              ' style="background:' +
              cellColor(value, matrix) +
              '">' +
              formatNumber(value) +
              "</td>"
            );
          })
          .join("");
        return (
          "<tr" +
          (rowIndex === selectedRow ? ' class="is-selected"' : "") +
          ">" +
          (rowLabels.length > 0
            ? '<th scope="row">' + rowLabels[rowIndex] + "</th>"
            : "") +
          cells +
          "</tr>"
        );
      })
      .join("");

    return (
      '<div class="rnn-live__matrix-scroll">' +
      '<table class="rnn-live__matrix">' +
      header +
      "<tbody>" +
      rows +
      "</tbody></table></div>"
    );
  }

  function matrixCard(title, matrix, note, active, options) {
    return (
      '<section class="rnn-live__matrix-card' +
      (active ? " rnn-live__matrix-card--active" : "") +
      '">' +
      '<div class="rnn-live__matrix-title">' +
      title +
      " · " +
      matrixShape(matrix) +
      "</div>" +
      matrixTable(matrix, options) +
      (note ? '<div class="rnn-live__matrix-note">' + note + "</div>" : "") +
      "</section>"
    );
  }

  function renderSignals(signals, currentTimeStep, selectedSignal) {
    const width = 780;
    const height = 255;
    const left = 92;
    const right = 22;
    const top = 16;
    const rowHeight = 45;
    const plotWidth = width - left - right;
    const amplitudeScale = 15;

    function xPosition(timeStep) {
      return left + (timeStep / (T - 1)) * plotWidth;
    }

    function yPosition(row, value) {
      return top + row * rowHeight + rowHeight / 2 - value * amplitudeScale;
    }

    let svg = (
      '<svg viewBox="0 0 ' +
      width +
      " " +
      height +
      '" role="img" aria-label="Five sine waves with the current RNN timestep highlighted">'
    );

    const cursorX = xPosition(currentTimeStep);
    svg +=
      '<rect class="rnn-live__cursor" x="' +
      (cursorX - 9) +
      '" y="5" width="18" height="' +
      (height - 20) +
      '" rx="5"></rect>';

    signals.forEach(function (signal, row) {
      const baseline = top + row * rowHeight + rowHeight / 2;
      const allPoints = signal
        .map(function (value, timeStep) {
          return xPosition(timeStep) + "," + yPosition(row, value);
        })
        .join(" ");
      const readPoints = signal
        .slice(0, currentTimeStep + 1)
        .map(function (value, timeStep) {
          return xPosition(timeStep) + "," + yPosition(row, value);
        })
        .join(" ");

      svg +=
        '<g class="rnn-live__signal-row' +
        (row === selectedSignal ? " is-selected" : "") +
        '" data-rnn-signal="' +
        row +
        '" tabindex="0" role="button" aria-label="Trace signal ' +
        row +
        ', class ' +
        LABELS[row] +
        '">';
      svg +=
        '<line class="rnn-live__svg-axis" x1="' +
        left +
        '" y1="' +
        baseline +
        '" x2="' +
        (width - right) +
        '" y2="' +
        baseline +
        '"></line>';
      svg +=
        '<text class="rnn-live__svg-label" x="6" y="' +
        (baseline + 4) +
        '">signal ' +
        row +
        " · y=" +
        LABELS[row] +
        "</text>";
      svg +=
        '<polyline class="rnn-live__wave" points="' +
        allPoints +
        '"></polyline>';
      svg +=
        '<polyline class="rnn-live__wave-read" points="' +
        readPoints +
        '"></polyline>';
      svg +=
        '<circle class="rnn-live__point" cx="' +
        cursorX +
        '" cy="' +
        yPosition(row, signal[currentTimeStep]) +
        '" r="4.5"></circle>';
      svg += "</g>";
    });

    svg +=
      '<text class="rnn-live__svg-label" x="' +
      cursorX +
      '" y="' +
      (height - 4) +
      '" text-anchor="middle">t=' +
      currentTimeStep +
      "</text>";
    svg += "</svg>";
    return svg;
  }

  function stageDetails(stage, timeStep) {
    if (stage === 0) {
      return {
        name: "Select the current samples",
        equation:
          "<strong>Select one column from the batch.</strong> " +
          "<code>x (5,10,1) → x[:, " +
          timeStep +
          ", :] (5,1)</code>. " +
          "The five rows are independent signals; each row contributes one amplitude.",
      };
    }
    if (stage === 1) {
      return {
        name: "Project the input into hidden space",
        equation:
          "<strong>Input path:</strong> " +
          "<code>x_t @ W_x: (5×1) @ (1×4) = (5×4)</code>. " +
          "The matching inner dimension is 1. One amplitude becomes four hidden features.",
      };
    }
    if (stage === 2) {
      return {
        name: "Transform the previous memory",
        equation:
          "<strong>Memory path:</strong> " +
          "<code>h_previous @ W_h: (5×4) @ (4×4) = (5×4)</code>. " +
          "The matching inner dimension is 4.",
      };
    }
    return {
      name: "Add, apply tanh, and update memory",
      equation:
        "<strong>State update:</strong> " +
        "<code>(5×4) + (5×4) + bias (4,) → pre (5×4) → tanh → h_" +
        timeStep +
        " (5×4)</code>. " +
        (timeStep < T - 1
          ? "This new state becomes h_previous at the next timestep."
          : "All ten samples are now summarized in h_final."),
    };
  }

  function renderPipeline(isFinal, stage, timeStep) {
    const nodes = [
      {
        title: "Select x_t",
        detail: "x[:, " + timeStep + ", :] · (5,1)",
      },
      {
        title: "Input path",
        detail: "x_t @ W_x · (5,4)",
      },
      {
        title: "Memory path",
        detail: "h_prev @ W_h · (5,4)",
      },
      {
        title: "Update memory",
        detail: "+ bias → tanh · (5,4)",
      },
      {
        title: "Classify",
        detail: "h_9 @ W.T · (5,2)",
      },
    ];
    const activeIndex = isFinal ? nodes.length - 1 : stage;

    return nodes
      .map(function (node, index) {
        const state =
          index === activeIndex
            ? " is-current"
            : index < activeIndex
              ? " is-complete"
              : "";
        const nodeHtml =
          '<div class="rnn-live__pipeline-node' +
          state +
          '"' +
          (index === activeIndex ? ' aria-current="step"' : "") +
          ">" +
          "<strong>" +
          node.title +
          "</strong><span>" +
          node.detail +
          "</span></div>";
        return (
          nodeHtml +
          (index < nodes.length - 1
            ? '<div class="rnn-live__pipeline-arrow" aria-hidden="true">→</div>'
            : "")
        );
      })
      .join("");
  }

  function renderTimeline(currentTimeStep, isFinal) {
    const timestepButtons = Array.from({ length: T }, function (_, timeStep) {
      const state = isFinal
        ? " is-complete"
        : timeStep === currentTimeStep
          ? " is-current"
          : timeStep < currentTimeStep
            ? " is-complete"
            : "";
      return (
        '<button type="button" class="' +
        state.trim() +
        '" data-rnn-jump-position="' +
        timeStep * OPERATIONS_PER_TIME_STEP +
        '"' +
        (!isFinal && timeStep === currentTimeStep
          ? ' aria-current="step"'
          : "") +
        ' aria-label="Jump to timestep ' +
        timeStep +
        '">t' +
        timeStep +
        "</button>"
      );
    }).join("");

    return (
      timestepButtons +
      '<button type="button" class="' +
      (isFinal ? "is-current" : "") +
      '" data-rnn-jump-position="' +
      FINAL_POSITION +
      '"' +
      (isFinal ? ' aria-current="step"' : "") +
      ">Classify</button>"
    );
  }

  function renderStageNavigation(currentTimeStep, stage, isFinal) {
    return STAGE_NAMES.map(function (name, stageIndex) {
      return (
        '<button type="button" class="' +
        (!isFinal && stageIndex === stage ? "is-current" : "") +
        '" data-rnn-jump-position="' +
        (currentTimeStep * OPERATIONS_PER_TIME_STEP + stageIndex) +
        '"' +
        (!isFinal && stageIndex === stage ? ' aria-current="step"' : "") +
        ">" +
        name +
        "</button>"
      );
    }).join("");
  }

  function renderSignalPicker(selectedSignal) {
    return SIGNAL_NAMES.map(function (name, signalIndex) {
      return (
        '<button type="button" class="' +
        (signalIndex === selectedSignal ? "is-current" : "") +
        '" data-rnn-select-signal="' +
        signalIndex +
        '"' +
        (signalIndex === selectedSignal ? ' aria-pressed="true"' : "") +
        ">" +
        name +
        " · y=" +
        LABELS[signalIndex] +
        "</button>"
      );
    }).join("");
  }

  function renderFeaturePicker(selectedFeature) {
    return HIDDEN_NAMES.map(function (name, featureIndex) {
      return (
        '<button type="button" class="' +
        (featureIndex === selectedFeature ? "is-current" : "") +
        '" data-rnn-select-feature="' +
        featureIndex +
        '"' +
        (featureIndex === selectedFeature ? ' aria-pressed="true"' : "") +
        ">" +
        name +
        "</button>"
      );
    }).join("");
  }

  function calculationRow(label, calculation, active) {
    const activeClass = active ? ' class="is-active"' : "";
    return (
      "<dt" +
      activeClass +
      ">" +
      label +
      "</dt><dd" +
      activeClass +
      ">" +
      calculation +
      "</dd>"
    );
  }

  function renderCellInspector(
    record,
    stage,
    selectedSignal,
    selectedFeature
  ) {
    const xValue = record.xT[selectedSignal][0];
    const inputWeight = MODEL.weightX[0][selectedFeature];
    const inputResult = record.inputProduct[selectedSignal][selectedFeature];
    const memoryTerms = record.hiddenPrevious[selectedSignal].map(
      function (hiddenValue, hiddenIndex) {
        return (
          formatPrecise(hiddenValue) +
          "×" +
          formatPrecise(MODEL.weightH[hiddenIndex][selectedFeature])
        );
      }
    );
    const memoryResult = record.memoryProduct[selectedSignal][selectedFeature];
    const bias = MODEL.bias[selectedFeature];
    const preActivation =
      record.preActivation[selectedSignal][selectedFeature];
    const hiddenValue = record.newHidden[selectedSignal][selectedFeature];

    return (
      '<div class="rnn-live__inspector-title">Exact scalar calculation · ' +
      SIGNAL_NAMES[selectedSignal] +
      " · t=" +
      record.timeStep +
      " · " +
      HIDDEN_NAMES[selectedFeature] +
      "</div>" +
      '<dl class="rnn-live__calculation">' +
      calculationRow(
        "Current x",
        formatPrecise(xValue),
        stage === 0
      ) +
      calculationRow(
        "Input term",
        formatPrecise(xValue) +
          "×" +
          formatPrecise(inputWeight) +
          " = " +
          formatPrecise(inputResult),
        stage === 1
      ) +
      calculationRow(
        "Memory dot",
        memoryTerms.join(" + ") + " = " + formatPrecise(memoryResult),
        stage === 2
      ) +
      calculationRow("Bias", formatPrecise(bias), false) +
      calculationRow(
        "Pre-activation",
        formatPrecise(inputResult) +
          " + " +
          formatPrecise(memoryResult) +
          " + " +
          formatPrecise(bias) +
          " = " +
          formatPrecise(preActivation),
        stage === 3
      ) +
      calculationRow(
        "New memory",
        "tanh(" +
          formatPrecise(preActivation) +
          ") = " +
          formatPrecise(hiddenValue),
        stage === 3
      ) +
      "</dl>"
    );
  }

  function renderFinalInspector(walkthrough, selectedSignal) {
    const hidden = walkthrough.finalHidden[selectedSignal];
    const classCalculations = Array.from({ length: C }, function (_, classIndex) {
      const terms = hidden.map(function (hiddenValue, hiddenIndex) {
        return (
          formatPrecise(hiddenValue) +
          "×" +
          formatPrecise(MODEL.classifierWeight[classIndex][hiddenIndex])
        );
      });
      return calculationRow(
        "Class " + classIndex + " logit",
        terms.join(" + ") +
          " + " +
          formatPrecise(MODEL.classifierBias[classIndex]) +
          " = " +
          formatPrecise(walkthrough.logits[selectedSignal][classIndex]),
        walkthrough.predictions[selectedSignal] === classIndex
      );
    }).join("");
    const target = LABELS[selectedSignal];
    const targetProbability =
      walkthrough.probabilities[selectedSignal][target];

    return (
      '<div class="rnn-live__inspector-title">Classifier calculation · ' +
      SIGNAL_NAMES[selectedSignal] +
      "</div>" +
      '<dl class="rnn-live__calculation">' +
      classCalculations +
      calculationRow(
        "Probabilities",
        "[" +
          walkthrough.probabilities[selectedSignal]
            .map(formatPrecise)
            .join(", ") +
          "]",
        false
      ) +
      calculationRow(
        "Decision",
        "target=" +
          target +
          ", prediction=" +
          walkthrough.predictions[selectedSignal],
        true
      ) +
      calculationRow(
        "Row loss",
        "−log(" +
          formatPrecise(targetProbability) +
          ") = " +
          formatPrecise(-Math.log(targetProbability)),
        false
      ) +
      "</dl>"
    );
  }

  function renderStageMatrices(
    record,
    stage,
    selectedSignal,
    selectedFeature
  ) {
    const batchOptions = {
      rowLabels: SIGNAL_NAMES,
      columnLabels: HIDDEN_NAMES,
      selectedRow: selectedSignal,
      selectedColumn: selectedFeature,
    };

    if (stage === 0) {
      return [
        matrixCard(
          "Current samples x_t",
          record.xT,
          "One amplitude from each of five signals.",
          true,
          {
            rowLabels: SIGNAL_NAMES,
            columnLabels: ["x"],
            selectedRow: selectedSignal,
          }
        ),
        matrixCard(
          "Previous hidden state",
          record.hiddenPrevious,
          record.timeStep === 0
            ? "All zeros at t=0."
            : "The h_t created at the previous timestep.",
          false,
          batchOptions
        ),
      ].join("");
    }

    if (stage === 1) {
      return [
        matrixCard("x_t", record.xT, "(5,1)", false, {
          rowLabels: SIGNAL_NAMES,
          columnLabels: ["x"],
          selectedRow: selectedSignal,
        }),
        matrixCard(
          "W_x",
          MODEL.weightX,
          "(1,4), shared across time",
          false,
          {
            rowLabels: ["x"],
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "x_t @ W_x",
          record.inputProduct,
          "(5,1) @ (1,4) = (5,4)",
          true,
          batchOptions
        ),
      ].join("");
    }

    if (stage === 2) {
      return [
        matrixCard(
          "h_previous",
          record.hiddenPrevious,
          "(5,4)",
          false,
          batchOptions
        ),
        matrixCard(
          "W_h",
          MODEL.weightH,
          "(4,4), shared across time",
          false,
          {
            rowLabels: HIDDEN_NAMES,
            columnLabels: HIDDEN_NAMES,
            selectedColumn: selectedFeature,
          }
        ),
        matrixCard(
          "h_previous @ W_h",
          record.memoryProduct,
          "(5,4) @ (4,4) = (5,4)",
          true,
          batchOptions
        ),
      ].join("");
    }

    return [
      matrixCard(
        "Input contribution",
        record.inputProduct,
        "(5,4)",
        false,
        batchOptions
      ),
      matrixCard(
        "Memory contribution",
        record.memoryProduct,
        "(5,4)",
        false,
        batchOptions
      ),
      matrixCard(
        "Bias",
        [MODEL.bias],
        "(4,) broadcast across five rows",
        false,
        {
          rowLabels: ["b"],
          columnLabels: HIDDEN_NAMES,
          selectedColumn: selectedFeature,
        }
      ),
      matrixCard(
        "Pre-activation",
        record.preActivation,
        "input + memory + bias",
        false,
        batchOptions
      ),
      matrixCard(
        "New hidden state h_t",
        record.newHidden,
        "tanh(pre), shape remains (5,4)",
        true,
        batchOptions
      ),
    ].join("");
  }

  function renderFinalMatrices(
    walkthrough,
    selectedSignal,
    selectedFeature
  ) {
    const predictedClass = walkthrough.predictions[selectedSignal];
    return [
      matrixCard(
        "Final hidden state h_9",
        walkthrough.finalHidden,
        "(5,4)",
        false,
        {
          rowLabels: SIGNAL_NAMES,
          columnLabels: HIDDEN_NAMES,
          selectedRow: selectedSignal,
          selectedColumn: selectedFeature,
        }
      ),
      matrixCard(
        "Classifier weight transpose",
        walkthrough.classifierWeightT,
        "PyTorch stores this as (2,4); multiplication uses (4,2).",
        false,
        {
          rowLabels: HIDDEN_NAMES,
          columnLabels: ["class 0", "class 1"],
          selectedColumn: predictedClass,
        }
      ),
      matrixCard(
        "Logits",
        walkthrough.logits,
        "(5,4) @ (4,2) + bias (2,) = (5,2)",
        true,
        {
          rowLabels: SIGNAL_NAMES,
          columnLabels: ["class 0", "class 1"],
          selectedRow: selectedSignal,
          selectedColumn: predictedClass,
        }
      ),
      matrixCard(
        "Softmax probabilities",
        walkthrough.probabilities,
        "Each row sums to 1.",
        false,
        {
          rowLabels: SIGNAL_NAMES,
          columnLabels: ["P(0)", "P(1)"],
          selectedRow: selectedSignal,
          selectedColumn: predictedClass,
        }
      ),
    ].join("");
  }

  function renderFinalResult(walkthrough) {
    return (
      '<div class="rnn-live__result-grid">' +
      '<div class="rnn-live__metric"><strong>Targets · shape (5,)</strong>' +
      "[" +
      LABELS.join(", ") +
      "]</div>" +
      '<div class="rnn-live__metric"><strong>Predictions · shape (5,)</strong>' +
      "[" +
      walkthrough.predictions.join(", ") +
      "]</div>" +
      '<div class="rnn-live__metric"><strong>Cross-entropy · scalar ()</strong>' +
      walkthrough.loss.toFixed(4) +
      "</div>" +
      '<div class="rnn-live__metric"><strong>Accuracy · scalar ()</strong>' +
      (100 * walkthrough.accuracy).toFixed(1) +
      "%</div>" +
      '<div class="rnn-live__metric"><strong>Confusion matrix · shape (2,2)</strong>' +
      "[[" +
      walkthrough.confusion[0].join(", ") +
      "], [" +
      walkthrough.confusion[1].join(", ") +
      "]]</div>" +
      "</div>" +
      '<div class="rnn-live__legend">' +
      "<span>Rows in the confusion matrix are true classes.</span>" +
      "<span>Columns are predicted classes.</span>" +
      "</div>"
    );
  }

  function initialize(container) {
    if (container.dataset.rnnReady === "true") return;
    container.dataset.rnnReady = "true";

    const signals = makeSignals();
    const walkthrough = computeWalkthrough(signals);

    const statusElement = container.querySelector("[data-rnn-status]");
    const signalElement = container.querySelector("[data-rnn-signals]");
    const equationElement = container.querySelector("[data-rnn-equation]");
    const inspectorElement = container.querySelector("[data-rnn-inspector]");
    const matrixElement = container.querySelector("[data-rnn-matrices]");
    const resultElement = container.querySelector("[data-rnn-result]");
    const progressElement = container.querySelector("[data-rnn-progress]");
    const timelineElement = container.querySelector("[data-rnn-timeline]");
    const stageNavElement = container.querySelector("[data-rnn-stage-nav]");
    const pipelineElement = container.querySelector("[data-rnn-pipeline]");
    const signalPickerElement = container.querySelector(
      "[data-rnn-signal-picker]"
    );
    const featurePickerElement = container.querySelector(
      "[data-rnn-feature-picker]"
    );
    const scrubber = container.querySelector("[data-rnn-scrubber]");
    const playButton = container.querySelector('[data-rnn-action="play"]');
    const previousButton = container.querySelector(
      '[data-rnn-action="previous"]'
    );
    const nextButton = container.querySelector('[data-rnn-action="next"]');
    const resetButton = container.querySelector('[data-rnn-action="reset"]');
    const speedControl = container.querySelector("[data-rnn-speed]");
    const toolbarTabs = Array.from(
      container.querySelectorAll("[data-rnn-toolbar-tab]")
    );
    const toolbarPanels = Array.from(
      container.querySelectorAll("[data-rnn-toolbar-panel]")
    );

    let position = 0;
    let timer = null;
    let selectedSignal = 0;
    let selectedFeature = 0;
    let activeToolbarTab = "steps";

    function activateToolbarTab(tabName, focusTab) {
      activeToolbarTab = tabName;

      toolbarTabs.forEach(function (tab) {
        const isActive = tab.dataset.rnnToolbarTab === activeToolbarTab;
        tab.classList.toggle("is-current", isActive);
        tab.setAttribute("aria-selected", String(isActive));
        tab.tabIndex = isActive ? 0 : -1;
        if (isActive && focusTab) tab.focus();
      });

      toolbarPanels.forEach(function (panel) {
        panel.hidden =
          panel.dataset.rnnToolbarPanel !== activeToolbarTab;
      });
    }

    function stop() {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
      playButton.textContent = "Play";
      playButton.setAttribute("aria-pressed", "false");
      statusElement.setAttribute("aria-live", "polite");
    }

    function render() {
      const isFinal = position === FINAL_POSITION;
      const timeStep = isFinal
        ? T - 1
        : Math.floor(position / OPERATIONS_PER_TIME_STEP);
      const stage = isFinal ? null : position % OPERATIONS_PER_TIME_STEP;

      progressElement.style.width =
        (((position + 1) / (FINAL_POSITION + 1)) * 100).toFixed(2) + "%";
      scrubber.value = String(position);
      previousButton.disabled = position === 0;
      nextButton.disabled = position === FINAL_POSITION;
      timelineElement.innerHTML = renderTimeline(timeStep, isFinal);
      stageNavElement.innerHTML = renderStageNavigation(
        timeStep,
        stage,
        isFinal
      );
      pipelineElement.innerHTML = renderPipeline(isFinal, stage, timeStep);
      signalPickerElement.innerHTML = renderSignalPicker(selectedSignal);
      featurePickerElement.innerHTML = renderFeaturePicker(selectedFeature);
      signalElement.innerHTML = renderSignals(
        signals,
        timeStep,
        selectedSignal
      );
      if (isFinal) {
        statusElement.textContent =
          "Final operation of " +
          (FINAL_POSITION + 1) +
          " · classify h_final";
        equationElement.innerHTML =
          "<strong>Classifier:</strong> " +
          "<code>h_final (5×4) @ W_classifier.T (4×2) + bias (2,) " +
          "→ logits (5×2)</code>. Then argmax gives predictions (5,).";
        inspectorElement.innerHTML = renderFinalInspector(
          walkthrough,
          selectedSignal
        );
        matrixElement.innerHTML = renderFinalMatrices(
          walkthrough,
          selectedSignal,
          selectedFeature
        );
        resultElement.hidden = false;
        resultElement.innerHTML = renderFinalResult(walkthrough);
        stop();
        return;
      }

      const details = stageDetails(stage, timeStep);
      statusElement.textContent =
        "Operation " +
        (position + 1) +
        " of " +
        (FINAL_POSITION + 1) +
        " · t=" +
        timeStep +
        " · " +
        details.name;
      equationElement.innerHTML = details.equation;
      inspectorElement.innerHTML = renderCellInspector(
        walkthrough.records[timeStep],
        stage,
        selectedSignal,
        selectedFeature
      );
      matrixElement.innerHTML = renderStageMatrices(
        walkthrough.records[timeStep],
        stage,
        selectedSignal,
        selectedFeature
      );
      resultElement.hidden = true;
      resultElement.innerHTML = "";
    }

    function jumpTo(nextPosition) {
      stop();
      position = Math.max(0, Math.min(FINAL_POSITION, nextPosition));
      render();
    }

    function play() {
      if (timer !== null) {
        stop();
        return;
      }
      if (position >= FINAL_POSITION) {
        position = 0;
        render();
      }
      playButton.textContent = "Pause";
      playButton.setAttribute("aria-pressed", "true");
      statusElement.setAttribute("aria-live", "off");
      timer = window.setInterval(function () {
        if (position < FINAL_POSITION) {
          position += 1;
          render();
        } else {
          stop();
        }
      }, Number(speedControl.value));
    }

    resetButton.addEventListener("click", function () {
      jumpTo(0);
    });

    previousButton.addEventListener("click", function () {
      jumpTo(position - 1);
    });

    nextButton.addEventListener("click", function () {
      jumpTo(position + 1);
    });

    playButton.addEventListener("click", play);

    speedControl.addEventListener("change", function () {
      if (timer !== null) {
        stop();
        play();
      }
    });

    scrubber.addEventListener("input", function () {
      jumpTo(Number(scrubber.value));
    });

    container.addEventListener("click", function (event) {
      const toolbarTab = event.target.closest("[data-rnn-toolbar-tab]");
      if (toolbarTab && container.contains(toolbarTab)) {
        activateToolbarTab(toolbarTab.dataset.rnnToolbarTab, false);
        return;
      }

      const jumpTarget = event.target.closest("[data-rnn-jump-position]");
      if (jumpTarget && container.contains(jumpTarget)) {
        jumpTo(Number(jumpTarget.dataset.rnnJumpPosition));
        return;
      }

      const signalTarget = event.target.closest(
        "[data-rnn-select-signal], [data-rnn-signal]"
      );
      if (signalTarget && container.contains(signalTarget)) {
        const nextSignal =
          signalTarget.dataset.rnnSelectSignal ??
          signalTarget.dataset.rnnSignal;
        selectedSignal = Number(nextSignal);
        stop();
        render();
        return;
      }

      const featureTarget = event.target.closest("[data-rnn-select-feature]");
      if (featureTarget && container.contains(featureTarget)) {
        selectedFeature = Number(featureTarget.dataset.rnnSelectFeature);
        stop();
        render();
      }
    });

    container.addEventListener("keydown", function (event) {
      const toolbarTab = event.target.closest("[data-rnn-toolbar-tab]");
      if (
        toolbarTab &&
        ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
      ) {
        event.preventDefault();
        const currentIndex = toolbarTabs.indexOf(toolbarTab);
        let nextIndex = currentIndex;

        if (event.key === "ArrowLeft") {
          nextIndex =
            (currentIndex - 1 + toolbarTabs.length) % toolbarTabs.length;
        } else if (event.key === "ArrowRight") {
          nextIndex = (currentIndex + 1) % toolbarTabs.length;
        } else if (event.key === "Home") {
          nextIndex = 0;
        } else if (event.key === "End") {
          nextIndex = toolbarTabs.length - 1;
        }

        activateToolbarTab(
          toolbarTabs[nextIndex].dataset.rnnToolbarTab,
          true
        );
        return;
      }

      const signalTarget = event.target.closest("[data-rnn-signal]");
      if (
        signalTarget &&
        (event.key === "Enter" || event.key === " ")
      ) {
        event.preventDefault();
        selectedSignal = Number(signalTarget.dataset.rnnSignal);
        stop();
        render();
        return;
      }

      if (event.target.matches("button, input, select")) return;

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        jumpTo(position - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        jumpTo(position + 1);
      } else if (event.key === "Home") {
        event.preventDefault();
        jumpTo(0);
      } else if (event.key === "End") {
        event.preventDefault();
        jumpTo(FINAL_POSITION);
      } else if (event.key === " ") {
        event.preventDefault();
        play();
      }
    });

    activateToolbarTab(activeToolbarTab, false);
    render();
  }

  function boot() {
    document.querySelectorAll("[data-rnn-live]").forEach(initialize);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
