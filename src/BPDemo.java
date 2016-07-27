import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;


public class BPDemo {
	
	public int numFeature, numHidden, numOutput;
	public int numOfSamples;
	public double[] activateInput, activateHidden, activateOutput;
	public double[][] weightInput, weightOutput;
	
	public double[][] dataTrains;
	public double[] targets;
	
	public static double[][] dataTest;
	public static double[] targetsTest;
	
	public BPDemo(int numFeature, int numOfSamples, int numHidden, int numOutput) {
		
		this.numFeature = numFeature;
		this.numHidden = numHidden;
		this.numOutput = numOutput;
		this.numOfSamples = numOfSamples;
		
		this.activateInput = new double[this.numFeature];
		this.activateHidden = new double[this.numHidden];
		this.activateOutput = new double[this.numOutput];
		
		this.weightInput = new double[this.numFeature][this.numHidden];
		this.weightOutput = new double[this.numHidden][this.numOutput];
				
		initializer();
		
	}
	
	public void initializer() {
		
		for (int i=0;i<this.numFeature; i++){
			this.activateInput[i] = 0.0;
		}
		
		for (int i=0;i<this.numHidden; i++){
			this.activateHidden[i] = 0.0;
		}

		for (int i=0;i<this.numOutput; i++){
			this.activateOutput[i] = 0.0;
		}

		for (int i=0;i<this.numFeature;i++) {
			for (int j=0;j<this.numHidden;j++) {
				this.weightInput[i][j] = Math.random();
			}
		}
		for (int i=0;i<this.numHidden;i++) {
			for (int j=0;j<this.numOutput;j++) {
				this.weightOutput[i][j] = Math.random();
			}
		}
	}
	
	public double sigmoid(double x) {
		return (1.0/(1.0 + Math.pow(Math.E, -x)));
	}
		
	public double update(double[] inputs) {
		
		for (int i=0; i<this.numFeature; i++) {
			this.activateInput[i] = inputs[i];
		}
		
		for (int j=0; j<this.numHidden; j++) {
			double sum = 0.0;
			for (int i=0; i<this.numFeature; i++) {
				sum += this.activateInput[i] * this.weightInput[i][j];
			}
			this.activateHidden[j] = sigmoid(sum);
		}
		
		for (int k=0; k<this.numOutput; k++) {
			double sum = 0.0;
			for (int j=0; j<this.numHidden; j++) {
				sum += this.activateHidden[j] * this.weightOutput[j][k];
			}
			this.activateOutput[k] = sigmoid(sum);
		}
		
		return this.activateOutput[0];
	}
	
	public double backPropagate(double[] targets, double learningRate) {
		
		double[] outputDeltas = new double[this.numOutput];
		for (int k=0; k<this.numOutput; k++) {
			double error = this.activateOutput[k] * (1.0 - this.activateOutput[k]) * (targets[0] - this.activateOutput[k]);
			outputDeltas[k] = error;
		}
		
		double[] hiddenDeltas = new double[this.numHidden];
		for (int j=0; j<this.numHidden; j++) {
			double error = 0.0;
			for (int k=0; k<this.numOutput; k++) {
				error += outputDeltas[k] * this.weightOutput[j][k];
			}
			error *= this.activateHidden[j] * (1 - this.activateHidden[j]);
			hiddenDeltas[j] = error;
		}
		
		for (int j=0;j<this.numHidden;j++) {
			for (int k=0;k<this.numOutput;k++) {
				double changeValues = outputDeltas[k] * this.activateHidden[j];
				this.weightOutput[j][k] += learningRate * changeValues ;
			}
		}
		
		for (int i=0;i<this.numFeature;i++) {
			for (int j=0;j<this.numHidden;j++) {
				double changeValues = hiddenDeltas[j] * this.activateInput[i];
				this.weightInput[i][j] += learningRate * changeValues ;
			}
		}
		
		double error = 0.0;
		for (int k=0; k<targets.length;k++) {
			error += 0.5 * Math.pow(targets[k] - this.activateOutput[k], 2);
		}
		
		return error;
	}
	
	public void train(double[][] dataTrain, double[] targets, int iter, double learningRate) {
		
		for (int i=0; i<iter; i++) {
			double error = 0.0;
			
			for (int p=0; p<this.numOfSamples; p++) {
				
				double[] inputs = dataTrain[p];
				double[] target = {targets[p]};
				
				update(inputs);
				error += backPropagate(target, learningRate);
				
			}
			
			if (i % 10 == 0) {
				System.out.println("Iteration " + i + "-th" + ", Error = " + error);
			}
		}
	}
	

	public void test(double[][] input, double[] expectedvalue) {
		
		double numOfCorrectSamples = 0;
		for (int p=0; p<expectedvalue.length; p++) {
			double predict = update(input[p]);
			int classPredict = predict > 0.5 ? 1 : 0;
			int classExpected = (int) expectedvalue[p];
			if	(classPredict == classExpected) {
				numOfCorrectSamples += 1;
			}
		}
		System.out.println("Correct rate = " + (double)numOfCorrectSamples/(double)expectedvalue.length);
	}
	
	public static void main(String[] args) {

//		BPDemo demo = new BPDemo(2, 10000, 16, 1);
//		demo.createDataForTrain(demo.numOfSamples);
//		demo.train(demo.dataTrains, demo.targets, 1000, 0.1);
//		
//		demo.createDataForTest(demo.numOfSamples);
//		demo.test(dataTest, targetsTest);		
	
	}
	

	public void createDataForTrain(int numOfSamples) {
		
		dataTrains = new double[numOfSamples][2];
		targets = new double[numOfSamples];
		
		for (int i=0; i<numOfSamples; i++) {
			dataTrains[i][0] = randDouble(-50.0, 50.0);
			dataTrains[i][1] = randDouble(-300.0, 300.0);
			double fvalue = sigmoid(f(dataTrains[i][0], dataTrains[i][1]));
			if (fvalue > 0.5) {
				targets[i] = 1.0;
			} else {
				targets[i] = 0.0;
			}
		}
	}
	
	public void createDataForTest(int numOfSamples) {

		dataTest = new double[numOfSamples][2];
		targetsTest = new double[numOfSamples];
		
		for (int i=0; i<numOfSamples; i++) {
			dataTest[i][0] = randDouble(1000.0, 2000.0);
			dataTest[i][1] = randDouble(500.0, 5000.0);
			double fvalue = sigmoid(f(dataTrains[i][0], dataTrains[i][1]));
			if (fvalue > 0.5) {
				targetsTest[i] = 1.0;
			} else {
				targetsTest[i] = 0.0;
			}
		}
	}

	public double f(double x, double y) {
		return 3*x*x - 2*y;
	}
	
	public static double randDouble(double rangeMin, double rangeMax) {
		Random r = new Random();
		double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
		return randomValue;
	}
	
}

