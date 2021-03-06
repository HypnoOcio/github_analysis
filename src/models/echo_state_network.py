import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class EchoStateRNNCell(keras.layers.Layer):
    """Echo-state RNN cell.
    """

    def __init__(self, units, decay=0.1, alpha=0.5, rho=1.0, sw=1.0, seed=None,
                 epsilon=None, sparseness=0.0,  activation=None, optimize=False,
                 optimize_vars=None, *args, **kwargs):
        """
        Args:
            units (int):  The number of units in the RNN cell.
            decay (float): Decay of the ODE of each unit. Default: 0.1.
            seed (int): seed for random numbers. Default None.
            epsilon (float): Discount from spectral radius 1. Default: 1e-10.
            alpha (float): [0,1], the proporsion of infinitesimal expansion vs infinitesimal rotation
                of the dynamical system defined by the inner weights
            sparseness (float): [0,1], sparseness of the inner weight matrix. Default: 0.
            rho (float): the scale of internal weights
            sw (float): the scale of input weights
            activation (callable): Nonlinearity to use.  Default: `tanh`.
            optimize (bool): whether to optimize variables (see optimize_Vars)
            optimize_vars (list): variables to be optimize ( default None -- all variable are trainable).
        """

        self.seed = seed
        self.units = units
        self.state_size = units
        self.sparseness = sparseness
        self.decay_ = decay
        self.alpha_ = alpha
        self.rho_ = rho
        self.sw_ = sw
        self.epsilon = epsilon
        self._activation = tf.tanh if activation is None else activation
        self.optimize = optimize
        self.optimize_vars = optimize_vars

        super(EchoStateRNNCell, self).__init__(*args, **kwargs)

    def build(self, input_shape):

        # alpha and rho default as tf non trainables
        self.optimize_table = {"alpha": False,
                               "rho": False,
                               "decay": False,
                               "sw": False}

        if self.optimize == True:
            # Set tf trainables
            for var in ["alpha", "rho", "decay", "sw"]:
                if var in self.optimize_vars:
                    self.optimize_table[var] = True
                else:
                    self.optimize_table[var] = False

        # leaky decay
        self.decay = tf.Variable(self.decay_, name="decay",
                                 dtype=tf.float32,
                                 trainable=self.optimize_table["decay"])
        # parameter for dynamic rotation/translation (0.5 means no modifications)
        self.alpha = tf.Variable(self.alpha_, name="alpha",
                                 dtype=tf.float32,
                                 trainable=self.optimize_table["alpha"])

        # the scale factor of the unitary spectral radius
        self.rho = tf.Variable(self.rho_, name="rho",
                               dtype=tf.float32,
                               trainable=self.optimize_table["rho"])

        # the scale factor of the input weights
        self.sw = tf.Variable(self.sw_, name="sw",
                              dtype=tf.float32,
                              trainable=self.optimize_table["sw"])
        
        self.alpha_store = tf.Variable(self.alpha_, name="alpha_store",
                             dtype=tf.float32, trainable=False) 
        
        self.echo_ratio = tf.Variable(1, name="echo_ratio",
                             dtype=tf.float32, trainable=False) 
                
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=keras.initializers.RandomUniform(-1, 1, seed=self.seed),
            name="kernel", trainable=False)

        self.recurrent_kernel_init = self.add_weight(
            shape=(self.units, self.units),
            initializer=keras.initializers.RandomNormal(seed=self.seed),
            name="recurrent_kernel", trainable=False)
       
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer=tf.zeros_initializer(),
            name="recurrent_kernel", trainable=False)
    
        self.recurrent_kernel_init.assign(self.setSparseness(self.recurrent_kernel_init))
        self.recurrent_kernel.assign(self.setAlpha(self.recurrent_kernel_init))
        self.echo_ratio.assign(self.echoStateRatio(self.recurrent_kernel))
        self.rho.assign(self.findEchoStateRho(self.recurrent_kernel*self.echo_ratio))
        
        self.built = True

    def setAlpha(self, W):
        W = 0.5*(self.alpha*(W + tf.transpose(W)) + (1 - self.alpha)*(W - tf.transpose(W)))
        return W

    def setSparseness(self, W):
        mask = tf.cast(tf.random.uniform(W.shape, seed=self.seed)
                       < (1 - self.sparseness), dtype=W.dtype)
        W = W * mask
        return W

    def echoStateRatio(self, W):
        eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
        return tf.reduce_max(tf.abs(eigvals))

    def findEchoStateRho(self, W):
        """Build the inner weight matrix initialixer W  so that
            1 - epsilon < rho(W)  < 1,
            where
            Wd = decay * W + (1 - decay) * I.
            See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.
            See also https://goo.gl/U6ALDd.
            Returns:
                A 2-D tensor representing the
                inner weights of an ESN
        """

        # Correct spectral radius for leaky units. The iteration
        #    has to reach this value
        target = 1.0
        # spectral radius and eigenvalues
        eigvals = tf.py_function(np.linalg.eigvals, [W], tf.complex64)
        x = tf.math.real(eigvals)
        y = tf.math.imag(eigvals)
        # solve quadratic equations
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        # just get the positive solutions
        sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)
        # and take the minor amongst them
        rho = tf.reduce_min(sol)
        return rho
  
    def clip_variables(self):
        """ clip parameters having been optimized to their limits
        """
        self.decay.assign(tf.clip_by_value(
            self.decay, 0.00000001, 0.25))
        self.alpha.assign(tf.clip_by_value(
            self.alpha, 0.000001, 0.9999999))
        self.rho.assign(tf.clip_by_value(
            self.rho, 0.5, 1.0e100))
        self.sw.assign(tf.clip_by_value(
            self.sw, 0.5, 1.0e100))
    
    def call(self, inputs, states):
        """ Echo-state RNN:
            x = x + h*(f(W*inp + U*g(x)) - x).
        """
        
        rkernel = self.setAlpha(self.recurrent_kernel_init)
        if self.alpha != self.alpha_store:
            self.clip_variables()
            self.echo_ratio.assign(self.echoStateRatio(rkernel))
            self.rho.assign(self.findEchoStateRho(rkernel*self.echo_ratio)) 
            self.alpha_store.assign(self.alpha)

        ratio = self.rho*self.echo_ratio*(1 - self.epsilon)

        prev_output = states[0]
        output = prev_output + self.decay*(
            tf.matmul(
                inputs,
                self.kernel * self.sw) +
            tf.matmul(
                self._activation(prev_output),
                rkernel*ratio)
            - prev_output)

        return self._activation(output), [output]