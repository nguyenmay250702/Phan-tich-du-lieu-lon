package map_reduce;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class KReducer extends Reducer<LongWritable, PointWritable, Text, Text> {

	private final Text newCentroidId = new Text();
	private final Text newCentroidValue = new Text();

	public void reduce(LongWritable centroidId, Iterable<PointWritable> partialSums, Context context) throws IOException, InterruptedException {
		
		Configuration conf = context.getConfiguration();
		Path outputDir = new Path("/BTL-KMean/map_reduce/nLoop-"+conf.get("nLoop")+"/3Reduce.txt");
		FileSystem fs = outputDir.getFileSystem(conf);
	
		FSDataOutputStream outputStream;
	
		if (fs.exists(outputDir)) {
		    outputStream = fs.append(outputDir);
		} else {
		    outputStream = fs.create(outputDir);
		}
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream));
		
		PointWritable ptFinalSum = PointWritable.copy(partialSums.iterator().next());
		writer.write("- intputReduce: \n\t+ centroidId = "+centroidId+ "\n\t+ partialSums = (" +ptFinalSum.toString() + ")");
		while (partialSums.iterator().hasNext()) {
			PointWritable P = partialSums.iterator().next();
			writer.write("; (" +P.toString() + ")");
			ptFinalSum.sum(P);
		}

		ptFinalSum.calcAverage();

		newCentroidId.set(centroidId.toString());
		newCentroidValue.set(ptFinalSum.toString());
		
		writer.write("\n- outputReduce: \n\t+ newCentroidId = "+ newCentroidId +"\n\t+ newCentroidValue = (" + newCentroidValue + ")\n\n");
		writer.close();
		
		context.write(newCentroidId, newCentroidValue);
	}
}