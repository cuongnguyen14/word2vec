import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Word2vec {

	public ArrayList<WordContext> wordCtx;
	public ArrayList<WordMapping> listMap;
	
	public Word2vec() {
		
	}
	
	public void train() {
		
	}
	
	public void prepareDataforTrain() {
		
		this.wordCtx = new ArrayList<>();
		this.listMap = new ArrayList<>();
		
		ArrayList<String> data = new ArrayList<String>();
		data.add("drink|juice|apple");data.add("eat|apple|orange");data.add("drink|juice|rice");
		data.add("drink|milk|juice");data.add("drink|rice|milk");data.add("drink|milk|water");
		data.add("orange|apple|juice");data.add("apple|drink|juice");data.add("rice|drink|milk");
		data.add("milk|water|drink");data.add("water|juice|drink");data.add("juice|water|drink");
		
		for	(String s : data) {
			String[] splitS = s.split("\\|", -1);
			WordContext wctx = new WordContext(splitS[2], Arrays.asList(splitS[0],splitS[1]));
			this.wordCtx.add(wctx);
			
			WordMapping map = new WordMapping(splitS[2], randomWeight(300));
			this.listMap.add(map);
		}

	}
	
	public double[] randomWeight(int dim) {
		double[] res = new double[dim];
		for (int i=0; i<dim; i++) {
			res[i] = randomDouble(0.0, 1.0);
		}
		return res;
	}
	
	public double randomDouble(double rangeMin, double rangeMax) {
		Random r = new Random();
		double randomValue = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
		return randomValue;
	}
	
	public static void main(String[] args) {

		Word2vec w2v = new Word2vec();
		w2v.prepareDataforTrain();
	
		BPDemo demo = new BPDemo(8, 12, 300, 8);
		
		demo.dataTrains = new double[12][300];
		demo.targets = new double[12];
		
		for (int i=0; i<12; i++) {
			demo.dataTrains[i] = w2v.randomWeight(300);
			demo.targets[i] = Math.pow(-1, i);
		}
		demo.train(demo.dataTrains, demo.targets, 1000, 0.1);
		
		int a = 0;
	}
	
	public double[][] dataTrains;
	public double[] targets;
	
}

/*
 * drink|juice|apple,eat|apple|orange,
 * drink|juice|rice,drink|milk|juice,
 * drink|rice|milk,drink|milk|water,
 * orange|apple|juice,apple|drink|juice,
 * rice|drink|milk,milk|water|drink,
 * water|juice|drink,juice|water|drink
 */
