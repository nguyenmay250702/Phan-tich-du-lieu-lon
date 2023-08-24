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
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class KCombiner extends Reducer<LongWritable, PointWritable, LongWritable, PointWritable> {

	public void reduce(LongWritable centroidId, Iterable<PointWritable> points, Context context) throws IOException, InterruptedException {
		
		Configuration conf = context.getConfiguration();
		Path outputDir = new Path("/BTL-KMean/map_reduce/nLoop-"+conf.get("nLoop")+"/2Combiner.txt");
		FileSystem fs = outputDir.getFileSystem(conf);
	
		FSDataOutputStream outputStream;
	
		if (fs.exists(outputDir)) {
		    outputStream = fs.append(outputDir);
		} else {
		    outputStream = fs.create(outputDir);
		}
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream));
		
		PointWritable ptSum = PointWritable.copy(points.iterator().next());
		writer.write("- intputCombiner: \n\t+ centroidId = "+centroidId+"\n\t+ points = ("+ ptSum.toString() + ")");
		while (points.iterator().hasNext()) {
			PointWritable P = points.iterator().next();
			writer.write("; ("+ P.toString() + ")");
			ptSum.sum(P);
		}
		writer.write("\n- outputCombiner: \n\t+ centroidId = " + centroidId);
		writer.write("\n\t+ ptSum = (" + ptSum.toString() + ")\n\n");
		writer.close();

		context.write(centroidId, ptSum);
	}
}