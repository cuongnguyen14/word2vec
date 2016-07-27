import java.util.List;

public class WordContext {
	public List<String> context;
	public String word;
	public int wordID;
	public WordContext(String word, List<String> context) {
		this.word = word;
		this.wordID = word.hashCode();
		this.context = context;
	}
}
