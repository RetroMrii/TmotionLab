import type {
    ConfusionPair,
} from "../types/api";


interface ConfusionTableProps {
    items: ConfusionPair[];
}


export function ConfusionTable({
    items,
}: ConfusionTableProps) {
    return (
        <div className="confusion-table-wrap">
            <table className="confusion-table">
                <thead>
                    <tr>
                        <th>
                            Benchmark
                        </th>

                        <th>
                            Predicted
                        </th>

                        <th>
                            Count
                        </th>

                        <th>
                            Mean score
                        </th>

                        <th>
                            Max score
                        </th>

                        <th>
                            High-score
                        </th>
                    </tr>
                </thead>

                <tbody>
                    {items.map(
                        (item) => (
                            <tr
                                key={
                                    item.confusion_pair
                                }
                            >
                                <td>
                                    {
                                        item
                                            .true_label_name
                                    }
                                </td>

                                <td>
                                    {
                                        item
                                            .textcnn_prediction
                                    }
                                </td>

                                <td>
                                    {item.count}
                                </td>

                                <td>
                                    {(
                                        item.mean_textcnn_score
                                        * 100
                                    ).toFixed(2)}
                                    %
                                </td>

                                <td>
                                    {(
                                        item.max_textcnn_score
                                        * 100
                                    ).toFixed(2)}
                                    %
                                </td>

                                <td>
                                    {
                                        item
                                            .high_score_errors
                                    }
                                </td>
                            </tr>
                        ),
                    )}
                </tbody>
            </table>
        </div>
    );
}