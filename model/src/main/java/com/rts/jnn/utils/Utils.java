package com.rts.jnn.utils;

import com.rts.jnn.core.activation.ActivationFunction;
import com.rts.jnn.core.activation.ReLUActivation;
import com.rts.jnn.core.activation.SigmoidActivation;
import com.rts.jnn.core.activation.TanhActivation;
import com.rts.jnn.core.initialization.InitializationFunction;
import com.rts.jnn.core.initialization.LeCunInitialization;
import com.rts.jnn.core.initialization.XavierInitialization;

public class Utils {

    public static ActivationFunction getActivationFunctionByName(String name) {
        return switch (name) {
            case "SigmoidActivation" -> new SigmoidActivation();
            case "ReLUActivation" -> new ReLUActivation();
            case "TanhActivation" -> new TanhActivation();
            default -> throw new IllegalArgumentException("Unknown activation function: " + name);
        };
    }

    public static InitializationFunction getInitializationFunctionByName(String name) {
        return switch (name) {
            case "XavierInitialization" -> new XavierInitialization();
            case "LeCunInitialization" -> new LeCunInitialization();
            default -> throw new IllegalArgumentException("Unknown initialization function: " + name);
        };
    }

}
