import 'package:flutter/material.dart';

class UserMessageTile extends StatelessWidget {
  final String message;
  const UserMessageTile({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('User'),
        Text(message),
      ],
    );
  }
}
