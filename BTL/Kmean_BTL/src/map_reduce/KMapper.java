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

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

public class KMapper extends Mapper<LongWritable, Text, LongWritable, PointWritable> {

	private PointWritable[] currCentroids;
	private final LongWritable centroidId = new LongWritable();
	private final PointWritable pointInput = new PointWritable();
	 
	@Override
	public void setup(Context context) {
		int nClusters = Integer.parseInt(context.getConfiguration().get("k"));

		this.currCentroids = new PointWritable[nClusters];
		for (int i = 0; i < nClusters; i++) {
			String[] centroid = context.getConfiguration().getStrings("C" + i);
			this.currCentroids[i] = new PointWritable(centroid);
		}
	}

	@Override
	protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		Path outputDir = new Path("/BTL-KMean/map_reduce/nLoop-"+conf.get("nLoop")+"/1Mapper.txt");
		Path PathOutputData = new Path("/BTL-KMean/data_output/data_output.txt");

		FileSystem fs0 = outputDir.getFileSystem(conf);
		FileSystem fs1 = PathOutputData.getFileSystem(conf);
	
		FSDataOutputStream outputStream;
		FSDataOutputStream outputStream1;
	
		if (fs0.exists(outputDir)||fs1.exists(PathOutputData)) {
		    outputStream = fs0.append(outputDir);
		    outputStream1 = fs1.append(PathOutputData);
		} else {
		    outputStream = fs0.create(outputDir);
		    outputStream1 = fs1.create(PathOutputData);
		}	
		
		BufferedWriter writer0 = new BufferedWriter(new OutputStreamWriter(outputStream));
		BufferedWriter writer1 = new BufferedWriter(new OutputStreamWriter(outputStream1));
		
		String[] arrPropPoint = value.toString().split(",");
		pointInput.set(arrPropPoint);
		
		writer0.write("-inputMapper: "+key.toString() + "; (" + pointInput.toString() + ")\n");
		
		double minDistance = Double.MAX_VALUE;
		int centroidIdNearest = 0;
		for (int i = 0; i < currCentroids.length; i++) {
			System.out.println("currCentroids[" + i + "]=" + currCentroids[i].toString());
			double distance = pointInput.calcDistance(currCentroids[i]);
			if (distance < minDistance) {
				centroidIdNearest = i;
				minDistance = distance;
			}
		}
		centroidId.set(centroidIdNearest);
		context.write(centroidId, pointInput); 
		
		writer0.write("-outputMapper: "+centroidId.toString() + "; (" + pointInput.toString() + ")\n\n");
		writer0.close();
		
		writer1.write(pointInput.toString() + ","+centroidId.toString()+"\n");
		writer1.close();

	}
}