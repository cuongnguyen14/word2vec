public class WordMapping {
	public String word;
	public double[] weightVector;
	public int wordID;
	public WordMapping(String word, double[] weightVector) {
		this.word = word;
		this.wordID = word.hashCode();
		this.weightVector = weightVector;
	}
}